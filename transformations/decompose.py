"""
Written by Juan Pablo Gutierrez
"""

from llama_index.core.llms.llm import LLM
from llama_index.core.indices.query.query_transform import StepDecomposeQueryTransform


def get_step_decompose_query_transform(llm: LLM = None, verbose: bool = False):
    """
    Get the step decompose query transform
    """
    return StepDecomposeQueryTransform(llm=llm, verbose=verbose)
