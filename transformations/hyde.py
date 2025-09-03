"""
Written by Juan Pablo Gutierrez
"""

from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.llms.llm import LLM
from llama_index.core.indices.query.query_transform import HyDEQueryTransform


def get_hyde_query_transform(
    llm: LLM = None,
    hyde_prompt: BasePromptTemplate = None,
    include_original: bool = True,
):
    """
    Get the HyDE reranker
    """
    return HyDEQueryTransform(
        llm=llm, hyde_prompt=hyde_prompt, include_original=include_original
    )
