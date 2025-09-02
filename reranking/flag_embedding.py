"""
Written by Juan Pablo Gutierrez
"""

from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

def get_flag_reranker(model_name: str, top_n: int = 3):
    """
    Get a FlagEmbeddingReranker instance.
    """

    return FlagEmbeddingReranker(
        top_n=top_n,
        model=model_name,
    )
