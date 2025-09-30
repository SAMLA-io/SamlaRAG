"""
Configuration Reader for the Declarative RAG Framework

This module handles reading, parsing, and validating configuration files
for building RAG pipelines.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from pydantic import BaseModel


class LLMConfig(BaseModel):
    """Configuration for LLM components."""

    type: str
    model: str
    temperature: Optional[float] = None


class QueryTransformConfig(BaseModel):
    """Configuration for query transform components."""

    type: str
    llm_config: Optional[LLMConfig] = None
    include_original: Optional[bool] = None
    verbose: Optional[bool] = None
    index_summary: Optional[str] = None


class PostprocessorConfig(BaseModel):
    """Configuration for postprocessor components."""

    type: str
    model: Optional[str] = None
    top_n: Optional[int] = None


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""

    type: str
    index_name: str
    dimension: Optional[int] = None
    metric: Optional[str] = None
    vector_type: Optional[str] = None
    region: Optional[str] = None
    cloud: Optional[str] = None


class RetrieverConfig(BaseModel):
    """Configuration for retriever."""

    similarity_top_k: int = 10


class QueryEngineConfig(BaseModel):
    """Configuration for query engine."""

    index_summary: Optional[str] = None


class RAGPipelineConfig(BaseModel):
    """Complete RAG pipeline configuration."""

    llm: LLMConfig
    query_transformers: List[QueryTransformConfig]
    postprocessors: List[PostprocessorConfig]
    vector_store: VectorStoreConfig
    retriever: RetrieverConfig
    query_engine: QueryEngineConfig


class ConfigReader(BaseModel):
    """Reads and validates RAG pipeline configuration from JSON files."""

    config_path: Path
    _config: Optional[RAGPipelineConfig] = None

    def __init__(self, config_path: str):
        super().__init__(config_path=Path(config_path))

    def load_config(self) -> RAGPipelineConfig:
        """Load and parse the configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)

        return self._parse_config(raw_config)

    def _parse_config(self, raw_config: Dict[str, Any]) -> RAGPipelineConfig:
        """Parse raw configuration into structured format."""
        # Parse LLMs
        llm_config = raw_config.get("llm", {})
        llm = LLMConfig(
            type=llm_config.get("type", "openai"),
            model=llm_config.get("model", "gpt-4o-mini"),
            temperature=llm_config.get("temperature"),
        )

        # Parse query transformers
        query_transformers = []
        for name, config in raw_config.get("query_transformers", {}).items():
            qt_config = QueryTransformConfig(
                type=config.get("type", name),  # Use name as type if not specified
                llm_config=LLMConfig(**config.get("llm")),
                include_original=config.get("include_original"),
                verbose=config.get("verbose"),
                index_summary=config.get("index_summary"),
            )
            query_transformers.append(qt_config)

        # Parse postprocessors
        postprocessors = []
        for name, config in raw_config.get("postprocessors", {}).items():
            # Determine type based on name or explicit type
            if "cohere" in name.lower():
                postprocessor_type = "cohere_rerank"
            elif "flag" in name.lower():
                postprocessor_type = "flag_rerank"
            else:
                postprocessor_type = config.get("type", "cohere_rerank")

            pp_config = PostprocessorConfig(
                type=postprocessor_type,
                model=config.get("model"),
                top_n=config.get("top_n", 3),
            )
            postprocessors.append(pp_config)

        # Parse vector store
        vector_store_config = raw_config.get("vector_store", {})
        vector_store = VectorStoreConfig(
            type=vector_store_config.get("type", "pinecone"),
            index_name=vector_store_config.get("index_name", "rag"),
            dimension=vector_store_config.get("dimension", 1536),
            metric=vector_store_config.get("metric", "cosine"),
            vector_type=vector_store_config.get("vector_type", "dense"),
            region=vector_store_config.get("region", "us-east-1"),
            cloud=vector_store_config.get("cloud", "aws"),
        )

        # Parse retriever
        retriever_config = raw_config.get("retriever", {})
        retriever = RetrieverConfig(
            similarity_top_k=retriever_config.get("similarity_top_k", 10)
        )

        # Parse query engine
        query_engine_config = raw_config.get("query_engine", {})
        query_engine = QueryEngineConfig(
            index_summary=query_engine_config.get("index_summary")
        )

        return RAGPipelineConfig(
            llm=llm,
            query_transformers=query_transformers,
            postprocessors=postprocessors,
            vector_store=vector_store,
            retriever=retriever,
            query_engine=query_engine,
        )

    def get_config(self) -> RAGPipelineConfig:
        """Get the loaded configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def validate_config(self) -> List[str]:
        """Validate the configuration and return any errors."""
        errors = []

        try:
            config = self.get_config()
        except Exception as e:  # pylint: disable=broad-exception-caught
            errors.append(f"Failed to load configuration: {str(e)}")
            return errors

        # Build lookup sets for names
        llm_names = set(getattr(llm, "name", None) for llm in config.llms)
        qt_names = set(getattr(qt, "name", None) for qt in config.query_transformers)

        # Validate LLM references in query transformers
        for qt in config.query_transformers:
            transform_config = qt
            qt_name = getattr(qt, "name", None)
            if transform_config.llm_config and transform_config.llm_config not in llm_names:
                errors.append(
                    f"Query transformer '{qt_name}' references unknown LLM '{transform_config.llm_config}'"
                )

        # Validate required fields
        if not config.vector_store.index_name:
            errors.append("Vector store index_name is required")

        return errors
