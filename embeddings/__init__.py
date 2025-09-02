"""
Written by Juan Pablo Gutierrez
"""
import os

from llama_index.embeddings.openai.base import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.vector_stores.simple import BasePydanticVectorStore

# Embed model should be defined in the config files (TODO: add later)
embed_model = OpenAIEmbedding(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
)


def get_pipeline(vector_store: BasePydanticVectorStore) -> IngestionPipeline:
    """
    Get the ingestion pipeline. Using a vector store to store the nodes.
    """
    return IngestionPipeline(
        transformations=[
            SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                include_metadata=True,
                embed_model=embed_model,
            ),
            embed_model,
        ],
        vector_store=vector_store,
    )
