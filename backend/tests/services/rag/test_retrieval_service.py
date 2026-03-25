# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.unit
class TestGetAllChunksFromKnowledgeBase:
    @pytest.mark.asyncio
    async def test_get_all_chunks_without_user_auth_check(self):
        """Internal all-chunks should work without passing a request user."""
        from app.services.rag.retrieval_service import RetrievalService

        kb = MagicMock()
        kb.id = 123
        kb.name = "KB"
        kb.namespace = "team-a"
        kb.user_id = 42
        kb.json = {
            "spec": {
                "retrievalConfig": {
                    "retriever_name": "retriever-a",
                    "retriever_namespace": "default",
                }
            }
        }

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = kb

        mock_backend = MagicMock()
        mock_backend.get_index_name.return_value = "kb-index"
        mock_backend.get_all_chunks.return_value = [
            {"content": "chunk", "title": "doc-1", "doc_ref": "1"}
        ]

        with patch(
            "app.services.rag.retrieval_service.retriever_kinds_service.get_retriever",
            return_value=MagicMock(),
        ):
            with patch(
                "app.services.rag.retrieval_service.create_storage_backend",
                return_value=mock_backend,
            ):
                result = await RetrievalService().get_all_chunks_from_knowledge_base(
                    knowledge_base_id=123,
                    db=db,
                    max_chunks=50,
                    query="debug query",
                )

        assert result == [{"content": "chunk", "title": "doc-1", "doc_ref": "1"}]
        mock_backend.get_index_name.assert_called_once_with("123", user_id=42)
        mock_backend.get_all_chunks.assert_called_once_with(
            knowledge_id="123",
            max_chunks=50,
            user_id=42,
        )


@pytest.mark.unit
class TestRetrieveForChatShell:
    @pytest.mark.asyncio
    async def test_auto_route_returns_direct_injection_records(self):
        """Backend should route to all-chunks when KB estimate fits context."""
        from app.services.rag.retrieval_service import RetrievalService

        db = MagicMock()

        with patch(
            "app.services.rag.retrieval_service.KnowledgeService.get_active_document_text_length_stats"
        ) as mock_stats:
            mock_stats.return_value = MagicMock(text_length_total=100)

            service = RetrievalService()
            service.get_all_chunks_from_knowledge_base = AsyncMock(
                return_value=[
                    {
                        "content": "chunk",
                        "title": "doc-1",
                        "doc_ref": "1",
                        "metadata": {"page": 1},
                    }
                ]
            )

            result = await service.retrieve_for_chat_shell(
                query="test",
                knowledge_base_ids=[123],
                db=db,
                max_results=5,
                context_window=10000,
                user_id=7,
            )

        assert result["mode"] == "direct_injection"
        assert result["total"] == 1
        assert result["records"][0]["score"] is None
        assert result["records"][0]["knowledge_base_id"] == 123

    @pytest.mark.asyncio
    async def test_auto_route_falls_back_to_rag_when_runtime_budget_is_insufficient(
        self,
    ):
        """Backend should own the final fit check when runtime budget is provided."""
        from app.services.rag.retrieval_service import RetrievalService

        db = MagicMock()

        with patch(
            "app.services.rag.retrieval_service.KnowledgeService.get_active_document_text_length_stats"
        ) as mock_stats:
            mock_stats.return_value = MagicMock(text_length_total=100)

            service = RetrievalService()
            service.get_all_chunks_from_knowledge_base = AsyncMock(
                return_value=[
                    {
                        "content": "This is a direct injection candidate chunk with enough text to exceed the runtime budget.",
                        "title": "doc-1",
                        "doc_ref": "1",
                        "metadata": {"page": 1},
                    }
                ]
            )
            service.retrieve_from_knowledge_base_internal = AsyncMock(
                return_value={
                    "records": [
                        {
                            "content": "retrieved",
                            "score": 0.9,
                            "title": "doc-1",
                            "metadata": {"page": 2},
                        }
                    ]
                }
            )

            result = await service.retrieve_for_chat_shell(
                query="test",
                knowledge_base_ids=[123],
                db=db,
                max_results=5,
                context_window=10000,
                available_injection_tokens=1,
                model_id="claude-3-5-sonnet",
                user_id=7,
            )

        assert result["mode"] == "rag_retrieval"
        assert result["records"][0]["score"] == 0.9
        assert result["records"][0]["knowledge_base_id"] == 123

    @pytest.mark.asyncio
    async def test_force_rag_route_uses_standard_retrieval(self):
        """Forced rag route should bypass direct injection candidate path."""
        from app.services.rag.retrieval_service import RetrievalService

        db = MagicMock()
        service = RetrievalService()
        service.retrieve_from_knowledge_base_internal = AsyncMock(
            return_value={
                "records": [
                    {
                        "content": "retrieved",
                        "score": 0.9,
                        "title": "doc-1",
                        "metadata": {"page": 2},
                    }
                ]
            }
        )

        result = await service.retrieve_for_chat_shell(
            query="test",
            knowledge_base_ids=[123],
            db=db,
            max_results=5,
            route_mode="rag_retrieval",
        )

        assert result["mode"] == "rag_retrieval"
        assert result["records"][0]["score"] == 0.9
        assert result["records"][0]["knowledge_base_id"] == 123
