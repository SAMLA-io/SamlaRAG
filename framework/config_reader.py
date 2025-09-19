"""
Configuration Reader for the Declarative RAG Framework

This module handles reading, parsing, and validating configuration files
for building RAG pipelines.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM components."""

    type: str
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class QueryTransformConfig:
    """Configuration for query transform components."""

    type: str
    llm: Optional[str] = None
    include_original: Optional[bool] = None
    verbose: Optional[bool] = None
    index_summary: Optional[str] = None


@dataclass
class PostprocessorConfig:
    """Configuration for postprocessor components."""

    type: str
    model: Optional[str] = None
    top_n: Optional[int] = None


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""

    type: str
    index_name: str
    dimension: Optional[int] = None
    metric: Optional[str] = None
    region: Optional[str] = None
    cloud: Optional[str] = None


@dataclass
class RetrieverConfig:
    """Configuration for retriever."""

    similarity_top_k: int = 10


@dataclass
class QueryEngineConfig:
    """Configuration for query engine."""

    index_summary: Optional[str] = None


@dataclass
class RAGPipelineConfig:
    """Complete RAG pipeline configuration."""

    llms: Dict[str, LLMConfig]
    query_transformers: Dict[str, QueryTransformConfig]
    postprocessors: Dict[str, PostprocessorConfig]
    vector_store: VectorStoreConfig
    retriever: RetrieverConfig
    query_engine: QueryEngineConfig


class ConfigReader:
    """Reads and validates RAG pipeline configuration from JSON files."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._config: Optional[RAGPipelineConfig] = None

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
        llms = {}
        for name, config in raw_config.get("llms", {}).items():
            llms[name] = LLMConfig(
                type=config.get("type", "openai"),
                model=config.get("model", "gpt-3.5-turbo"),
                temperature=config.get("temperature"),
                max_tokens=config.get("max_tokens"),
            )

        # Parse query transformers
        query_transformers = {}
        for name, config in raw_config.get("query_transformers", {}).items():
            query_transformers[name] = QueryTransformConfig(
                type=config.get("type", name),  # Use name as type if not specified
                llm=config.get("llm"),
                include_original=config.get("include_original"),
                verbose=config.get("verbose"),
                index_summary=config.get("index_summary"),
            )

        # Parse postprocessors
        postprocessors = {}
        for name, config in raw_config.get("postprocessors", {}).items():
            # Determine type based on name or explicit type
            if "cohere" in name.lower():
                postprocessor_type = "cohere_rerank"
            elif "flag" in name.lower():
                postprocessor_type = "flag_rerank"
            else:
                postprocessor_type = config.get("type", "cohere_rerank")

            postprocessors[name] = PostprocessorConfig(
                type=postprocessor_type,
                model=config.get("model"),
                top_n=config.get("top_n", 3),
            )

        # Parse vector store
        vector_store_config = raw_config.get("vector_store", {})
        vector_store = VectorStoreConfig(
            type=vector_store_config.get("type", "pinecone"),
            index_name=vector_store_config.get("index_name", "rag"),
            dimension=vector_store_config.get("dimension", 1536),
            metric=vector_store_config.get("metric", "cosine"),
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
            llms=llms,
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

        # Validate LLM references in query transformers
        for name, transform_config in config.query_transformers.items():
            if transform_config.llm and transform_config.llm not in config.llms:
                errors.append(
                    f"Query transformer '{name}' references unknown LLM '{transform_config.llm}'"
                )

        # Validate required fields
        if not config.vector_store.index_name:
            errors.append("Vector store index_name is required")

        return errors
