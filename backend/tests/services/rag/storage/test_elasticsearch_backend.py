# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ElasticsearchBackend get_all_chunks diagnostics."""

from unittest.mock import MagicMock, patch


class TestGetAllChunks:
    """Tests for ElasticsearchBackend.get_all_chunks."""

    @patch("app.services.rag.storage.elasticsearch_backend.Elasticsearch")
    def test_get_all_chunks_returns_parsed_chunks_and_logs_summary(
        self, mock_client_class, caplog
    ):
        """Should parse hits and log a useful summary for debugging."""
        from app.services.rag.storage.elasticsearch_backend import ElasticsearchBackend

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.indices.exists.return_value = True
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 1, "relation": "eq"},
                "hits": [
                    {
                        "_source": {
                            "content": "chunk content",
                            "metadata": {
                                "source_file": "doc-a.md",
                                "chunk_index": 3,
                                "doc_ref": "doc_1",
                                "knowledge_id": "kb_1",
                            },
                        }
                    }
                ],
            }
        }

        backend = ElasticsearchBackend(
            {
                "url": "http://localhost:9200",
                "indexStrategy": {"mode": "per_dataset", "prefix": "test"},
            }
        )

        with caplog.at_level("INFO"):
            result = backend.get_all_chunks(knowledge_id="kb_1", max_chunks=100)

        assert len(result) == 1
        assert result[0]["doc_ref"] == "doc_1"
        assert result[0]["chunk_id"] == 3
        assert "get_all_chunks search completed" in caplog.text
        assert "get_all_chunks parsed 1 chunks" in caplog.text

    @patch("app.services.rag.storage.elasticsearch_backend.Elasticsearch")
    def test_get_all_chunks_logs_samples_when_term_query_returns_empty(
        self, mock_client_class, caplog
    ):
        """Should log index samples when the knowledge_id term query returns no hits."""
        from app.services.rag.storage.elasticsearch_backend import ElasticsearchBackend

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.indices.exists.return_value = True
        mock_client.search.side_effect = [
            {"hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}},
            {
                "hits": {
                    "hits": [
                        {
                            "_source": {
                                "metadata": {
                                    "knowledge_id": "other_kb",
                                    "doc_ref": "doc_x",
                                    "chunk_index": 7,
                                    "source_file": "sample.txt",
                                }
                            }
                        }
                    ]
                }
            },
        ]
        mock_client.count.return_value = {"count": 5}

        backend = ElasticsearchBackend(
            {
                "url": "http://localhost:9200",
                "indexStrategy": {"mode": "per_dataset", "prefix": "test"},
            }
        )

        with caplog.at_level("WARNING"):
            result = backend.get_all_chunks(knowledge_id="kb_1", max_chunks=100)

        assert result == []
        assert "get_all_chunks returned empty" in caplog.text
        assert "index_doc_count=5" in caplog.text
        assert "other_kb" in caplog.text
