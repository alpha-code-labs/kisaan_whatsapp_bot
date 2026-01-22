import os
import pytest

from services.rag_builder import retrieve_rag_evidence, warm_rag_cache


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY is required for embeddings",
)
def test_rag_retrieval_smoke():
    assert warm_rag_cache() is True

    queries = [
        "Wheat | What is the recommended sowing time for wheat in Haryana?",
    ]
    results = retrieve_rag_evidence(queries, top_k=1)

    assert isinstance(results, list)
    assert len(results) == 1
    assert "status" in results[0]
