# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
State machine helpers for knowledge document indexing.

This module owns the business-level idempotency rules for document indexing:
- prevent duplicate enqueue while a generation is already queued/running
- version each indexing attempt with index_generation
- reject stale Celery redelivery/retry tasks for old generations
- update terminal state only when the task still matches the active generation
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.knowledge import DocumentIndexStatus, DocumentStatus, KnowledgeDocument


@dataclass(frozen=True)
class IndexEnqueueDecision:
    """Decision returned before sending a Celery indexing task."""

    should_enqueue: bool
    generation: Optional[int]
    reason: str
    previous_status: Optional[DocumentIndexStatus] = None


@dataclass(frozen=True)
class IndexExecutionDecision:
    """Decision returned when a worker starts processing a task."""

    should_execute: bool
    reason: str


ACTIVE_INDEX_STATUSES = {
    DocumentIndexStatus.QUEUED,
    DocumentIndexStatus.INDEXING,
}


def _utcnow() -> datetime:
    """Return a timezone-naive UTC timestamp for DB comparisons."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _get_active_index_stale_reason(
    document: KnowledgeDocument,
) -> Optional[str]:
    """Return a stale reason when an active indexing state is expired."""
    if document.updated_at is None:
        return None

    age_seconds = (_utcnow() - document.updated_at).total_seconds()
    if (
        document.index_status == DocumentIndexStatus.QUEUED
        and age_seconds >= settings.KNOWLEDGE_INDEX_STALE_QUEUED_SECONDS
    ):
        return "stale_queued"

    if (
        document.index_status == DocumentIndexStatus.INDEXING
        and age_seconds >= settings.KNOWLEDGE_INDEX_STALE_INDEXING_SECONDS
    ):
        return "stale_indexing"

    return None


def get_document_index_lock_name(document_id: int) -> str:
    """Return the Redis lock name for a document indexing task."""
    return f"knowledge:index_document:{document_id}"


def prepare_document_index_enqueue(
    db: Session,
    document_id: int,
    *,
    allow_if_success: bool = False,
    replace_active: bool = False,
) -> IndexEnqueueDecision:
    """
    Prepare a document for a new indexing generation.

    This function is called before sending a Celery task. It updates the
    business state in the database so later duplicate requests can be skipped.
    """
    document = (
        db.query(KnowledgeDocument)
        .filter(KnowledgeDocument.id == document_id)
        .with_for_update()
        .first()
    )
    if document is None:
        return IndexEnqueueDecision(
            should_enqueue=False,
            generation=None,
            reason="document_not_found",
        )

    current_status = document.index_status or DocumentIndexStatus.NOT_INDEXED

    if current_status in ACTIVE_INDEX_STATUSES and not replace_active:
        stale_reason = _get_active_index_stale_reason(document)
        if stale_reason is None:
            db.rollback()
            return IndexEnqueueDecision(
                should_enqueue=False,
                generation=document.index_generation,
                reason="already_in_progress",
                previous_status=current_status,
            )

        next_generation = (document.index_generation or 0) + 1
        document.index_generation = next_generation
        document.index_status = DocumentIndexStatus.QUEUED
        db.commit()

        return IndexEnqueueDecision(
            should_enqueue=True,
            generation=next_generation,
            reason="scheduled_after_stale_recovery",
            previous_status=current_status,
        )

    if current_status == DocumentIndexStatus.SUCCESS and not allow_if_success:
        db.rollback()
        return IndexEnqueueDecision(
            should_enqueue=False,
            generation=document.index_generation,
            reason="already_indexed",
            previous_status=current_status,
        )

    next_generation = (document.index_generation or 0) + 1
    document.index_generation = next_generation
    document.index_status = DocumentIndexStatus.QUEUED

    db.commit()

    return IndexEnqueueDecision(
        should_enqueue=True,
        generation=next_generation,
        reason="scheduled",
        previous_status=current_status,
    )


def mark_document_index_enqueue_failed(
    db: Session,
    document_id: int,
    generation: int,
) -> bool:
    """Mark a queued generation as failed when broker dispatch fails."""
    updated = (
        db.query(KnowledgeDocument)
        .filter(
            KnowledgeDocument.id == document_id,
            KnowledgeDocument.index_generation == generation,
            KnowledgeDocument.index_status == DocumentIndexStatus.QUEUED,
        )
        .update(
            {
                KnowledgeDocument.index_status: DocumentIndexStatus.FAILED,
                KnowledgeDocument.updated_at: _utcnow(),
            },
            synchronize_session=False,
        )
    )
    db.commit()
    return updated > 0


def mark_document_index_started(
    db: Session,
    document_id: int,
    generation: int,
) -> IndexExecutionDecision:
    """Transition a queued generation into indexing state."""
    document = (
        db.query(KnowledgeDocument)
        .filter(KnowledgeDocument.id == document_id)
        .with_for_update()
        .first()
    )
    if document is None:
        db.rollback()
        return IndexExecutionDecision(
            should_execute=False,
            reason="document_not_found",
        )

    if document.index_generation != generation:
        db.rollback()
        return IndexExecutionDecision(
            should_execute=False,
            reason="stale_generation",
        )

    current_status = document.index_status or DocumentIndexStatus.NOT_INDEXED
    if current_status == DocumentIndexStatus.SUCCESS:
        db.rollback()
        return IndexExecutionDecision(
            should_execute=False,
            reason="already_completed",
        )

    if current_status == DocumentIndexStatus.NOT_INDEXED:
        db.rollback()
        return IndexExecutionDecision(
            should_execute=False,
            reason="not_scheduled",
        )

    if current_status == DocumentIndexStatus.FAILED:
        db.rollback()
        return IndexExecutionDecision(
            should_execute=False,
            reason="already_failed",
        )

    document.index_status = DocumentIndexStatus.INDEXING
    db.commit()

    return IndexExecutionDecision(
        should_execute=True,
        reason="started",
    )


def mark_document_index_succeeded(
    db: Session,
    document_id: int,
    generation: int,
    *,
    chunks: Optional[dict] = None,
    chunk_storage_enabled: bool = False,
) -> bool:
    """Persist a successful indexing result for the active generation."""
    update_payload = {
        KnowledgeDocument.index_status: DocumentIndexStatus.SUCCESS,
        KnowledgeDocument.is_active: True,
        KnowledgeDocument.status: DocumentStatus.ENABLED,
    }

    if chunk_storage_enabled:
        update_payload[KnowledgeDocument.chunks] = chunks

    updated = (
        db.query(KnowledgeDocument)
        .filter(
            KnowledgeDocument.id == document_id,
            KnowledgeDocument.index_generation == generation,
            KnowledgeDocument.index_status.in_(ACTIVE_INDEX_STATUSES),
        )
        .update(
            {
                **update_payload,
                KnowledgeDocument.updated_at: _utcnow(),
            },
            synchronize_session=False,
        )
    )
    db.commit()
    return updated > 0


def mark_document_index_failed(
    db: Session,
    document_id: int,
    generation: int,
) -> bool:
    """Persist a failed indexing result for the active generation."""
    updated = (
        db.query(KnowledgeDocument)
        .filter(
            KnowledgeDocument.id == document_id,
            KnowledgeDocument.index_generation == generation,
            KnowledgeDocument.index_status.in_(ACTIVE_INDEX_STATUSES),
        )
        .update(
            {
                KnowledgeDocument.index_status: DocumentIndexStatus.FAILED,
                KnowledgeDocument.updated_at: _utcnow(),
            },
            synchronize_session=False,
        )
    )
    db.commit()
    return updated > 0
