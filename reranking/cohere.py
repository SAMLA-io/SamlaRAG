"""
Written by Juan Pablo Gutierrez
"""

from llama_index.postprocessor.cohere_rerank import CohereRerank

def get_cohere_reranker(model_name: str, top_n: int=3):
    """
    Get a CohereReranker instance.
    """
    return CohereRerank(model=model_name, top_n=top_n)
