"""RAG retrieval tests."""

from src.agents.rag import RAGRetriever, RetrievalResult


def test_retrieve_returns_top_chunks() -> None:
    retriever = RAGRetriever()
    result = retriever.retrieve("sports outdoor discount", top_k=2)
    assert isinstance(result, RetrievalResult)
    assert len(result.chunks) == 2
    assert len(result.sources) == 2
    assert len(result.scores) == 2


def test_retrieve_prefers_relevant_document() -> None:
    retriever = RAGRetriever()
    result = retriever.retrieve("sports outdoor campaign", top_k=1)
    assert "sports and outdoor" in result.chunks[0]
    assert result.scores[0] > 0


def test_scores_are_sorted_descending() -> None:
    retriever = RAGRetriever()
    result = retriever.retrieve("decoration kitchen", top_k=4)
    assert result.scores == sorted(result.scores, reverse=True)


def test_custom_documents() -> None:
    docs = [{"id": "x", "text": "Coffee machines on sale", "source": "catalog"}]
    retriever = RAGRetriever(documents=docs)
    result = retriever.retrieve("coffee", top_k=5)
    assert "Coffee machines" in result.chunks[0]


def test_empty_query_returns_top_k() -> None:
    retriever = RAGRetriever()
    result = retriever.retrieve("", top_k=3)
    assert len(result.chunks) == 3
