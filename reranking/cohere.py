"""
Written by Juan Pablo Gutierrez
"""

from cohere import V2RerankResponse
from reranking import co


def rerank(model: str, query: str, documents: list[str]) -> V2RerankResponse:
    """
    Rerank the documents using Cohere.

    Args:
        model: The model to use for reranking.
        query: The query to use for reranking.
        documents: The documents to use for reranking.

    Returns:
        A list of reranked documents.
    """
    return co.rerank(model=model, query=query, documents=documents, top_n=5)
