"""
Written by Juan Pablo Gutierrez

This module is responsible for building a RAG pipeline from a configuration file.
"""

from typing import Optional, List
from pydantic import BaseModel
from llama_index.llms.openai import OpenAI
from llama_index.core.llms.llm import LLM
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.indices.query.query_transform.base import BaseQueryTransform
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.query_engine.multistep_query_engine import MultiStepQueryEngine
from pinecone.db_control.models import ServerlessSpec
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine

from transformations.decompose import StepDecomposeQueryTransform
from transformations.decompose import get_step_decompose_query_transform
from transformations.hyde import get_hyde_query_transform
from framework.config_reader import ConfigReader, RAGPipelineConfig, LLMConfig
from reranking.cohere import get_cohere_reranker
from reranking.flag_embedding import get_flag_reranker
from vector_database.index import list_indexes, create_index


class RAGPipeline(BaseModel):
    """A RAG pipeline."""


class RAGBuilder(BaseModel):
    """Builds a RAG pipeline from a configuration file."""

    rag_pipeline_config: RAGPipelineConfig
    _vector_store: Optional[VectorStoreIndex] = None
    _vector_index: Optional[VectorStoreIndex] = None
    _retriever: Optional[BaseRetriever] = None
    _query_transformers: Optional[list[BaseQueryTransform]] = None
    _postprocessors: Optional[List[BaseNodePostprocessor]] = None

    def __init__(self, config_path: str):
        rag_pipeline_config = ConfigReader(config_path).get_config()
        super().__init__(rag_pipeline_config=rag_pipeline_config)

    def build_pipeline(self):
        """Builds a RAG pipeline from a configuration file."""
        self._build_llm(self.rag_pipeline_config.llm)
        self._build_query_transformers()
        self._build_postprocessors()
        self._build_vector_store()
        self._build_retriever()
        return self._build_query_engine()

    def _build_llm(self, llm_config: Optional[LLMConfig]) -> LLM:
        """
        Builds an LLM from a configuration object.
        If the LLM config is not provided, it uses the LLM from the pipeline config.
        """

        if llm_config is None:
            llm_config = self.rag_pipeline_config.llm

        match llm_config.type:
            case "openai":
                return OpenAI(
                    model=llm_config.model,
                    temperature=llm_config.temperature,
                )
            case _:
                raise ValueError(f"Unsupported LLM type: {llm_config.type}")

    def _build_query_transformers(self) -> list[BaseQueryTransform]:
        """Builds the query transformers from the configuration object."""

        if self._query_transformers is not None:
            return self._query_transformers

        query_transformers = []
        for qt in self.rag_pipeline_config.query_transformers:

            # Build LLM for query transformer from config if available, otherwise use pipeline LLM
            llm: LLM = None
            if qt.llm_config:
                llm = self._build_llm(qt.llm_config)
            else:
                llm = self._build_llm(self.rag_pipeline_config.llm)

            match qt.type:
                case "hyde":
                    query_transformers.append(get_hyde_query_transform(llm))
                case "step_decompose":
                    query_transformers.append(
                        get_step_decompose_query_transform(llm, verbose=True)
                    )
                case _:
                    raise ValueError(f"Unsupported query transformer type: {qt.type}")

        self._query_transformers = query_transformers
        return query_transformers

    def _build_postprocessors(self) -> List[BaseNodePostprocessor]:
        """Builds the postprocessors from the configuration object."""

        if self._postprocessors is not None:
            return self._postprocessors

        postprocessors: List[BaseNodePostprocessor] = []
        for pp in self.rag_pipeline_config.postprocessors:
            match pp.type:
                case "cohere_rerank":
                    postprocessors.append(
                        get_cohere_reranker(model_name=pp.model, top_n=pp.top_n)
                    )
                case "flag_rerank":
                    postprocessors.append(
                        get_flag_reranker(model_name=pp.model, top_n=pp.top_n)
                    )
                case _:
                    raise ValueError(f"Unsupported postprocessor type: {pp.type}")

        self._postprocessors = postprocessors
        return postprocessors

    def _build_vector_store(self) -> VectorStoreIndex:
        """Builds the vector store from the configuration object."""

        if self._vector_store is not None:
            return self._vector_store

        match self.rag_pipeline_config.vector_store.type:
            case "pinecone":

                indexes = list_indexes()
                if not any(
                    index.name == self.rag_pipeline_config.vector_store.index_name
                    for index in indexes
                ):
                    create_index(
                        index_name=self.rag_pipeline_config.vector_store.index_name,
                        dimension=self.rag_pipeline_config.vector_store.dimension,
                        metric=self.rag_pipeline_config.vector_store.metric,
                        vector_type=self.rag_pipeline_config.vector_store.vector_type,
                        spec=ServerlessSpec(
                            region=self.rag_pipeline_config.vector_store.region,
                            cloud=self.rag_pipeline_config.vector_store.cloud,
                        ),
                    )

                vector_store = PineconeVectorStore(
                    index_name=self.rag_pipeline_config.vector_store.index_name
                )
                self._vector_store = vector_store
                return vector_store
            case _:
                raise ValueError(
                    f"Unsupported vector store type: {self.rag_pipeline_config.vector_store.type}"
                )

    def _build_vector_index(self) -> VectorStoreIndex:
        """Builds the vector index from the configuration object."""
        if self._vector_index is not None:
            return self._vector_index

        vector_store = self._build_vector_store()
        self._vector_index = VectorStoreIndex.from_vector_store(vector_store)
        return self._vector_index

    def _build_retriever(self) -> BaseRetriever:
        """Builds the retriever from the configuration object."""

        if self._retriever is not None:
            return self._retriever

        self._retriever = VectorIndexRetriever(
            index=self._build_vector_index(),
            similarity_top_k=self.rag_pipeline_config.retriever.similarity_top_k,
        )
        return self._retriever

    def _build_query_engine(self) -> BaseQueryEngine:
        """Builds the query engine from the configuration object."""

        query_transformers = self._build_query_transformers()

        if len(query_transformers) > 1:
            retriever = self._build_retriever()
            index = self._build_vector_index()
            base_query_engine = RetrieverQueryEngine(
                retriever=retriever,
                node_postprocessors=self._build_postprocessors(),
            )

            hyde_transform = next(
                (qt for qt in query_transformers if isinstance(qt, HyDEQueryTransform)),
                None,
            )

            hyde_query_engine = TransformQueryEngine(
                query_engine=base_query_engine, query_transform=hyde_transform
            )

            step_decompose_query_transform = next(
                (
                    qt
                    for qt in query_transformers
                    if isinstance(qt, StepDecomposeQueryTransform)
                ),
                None,
            )

            return MultiStepQueryEngine(
                query_engine=hyde_query_engine,
                query_transform=step_decompose_query_transform,
                index_summary=self.rag_pipeline_config.query_engine.index_summary,
            )
        else:
            index = self._build_vector_index()
            return index.as_query_engine(
                llm=self._build_llm(self.rag_pipeline_config.llm),
                index_summary=self.rag_pipeline_config.query_engine.index_summary,
            )
