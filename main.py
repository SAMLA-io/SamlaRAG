"""
Written by Juan Pablo Gutierrez
"""

from pinecone.db_control.models import ServerlessSpec
from pinecone.db_control.enums import Metric, VectorType
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.query_engine import (
    RetrieverQueryEngine,
    TransformQueryEngine,
    MultiStepQueryEngine,
)
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from dotenv import load_dotenv
from reranking.cohere import get_cohere_reranker
from reranking.flag_embedding import get_flag_reranker
from vector_database.index import create_index, list_indexes
from vector_database import pc
from transformations.hyde import get_hyde_query_transform
from transformations.decompose import get_step_decompose_query_transform

load_dotenv()

print("Existing indexes:")
indexes = list_indexes()
for index in indexes:
    print(f"  - {index.name}")

INDEX_NAME = "rag"
if not any(index.name == INDEX_NAME for index in indexes):
    print(f"Creating index '{INDEX_NAME}'...")
    create_index(
        index_name=INDEX_NAME,
        dimension=1536,
        metric=Metric.COSINE,
        vector_type=VectorType.DENSE,
        spec=ServerlessSpec(region="us-east-1", cloud="aws"),
    )
    print(f"Index '{INDEX_NAME}' created successfully!")
else:
    print(f"Index '{INDEX_NAME}' already exists.")

vector_store = PineconeVectorStore(pinecone_index=pc.Index(INDEX_NAME))
vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# This is the most important part of the RAG
retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10)

node_postprocessor = [
    get_cohere_reranker(model_name="rerank-v3.5", top_n=3),
    get_flag_reranker(model_name="BAAI/bge-reranker-base", top_n=3),
]

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=node_postprocessor,
)

hyde = get_hyde_query_transform(include_original=True)
hyde_query_engine = TransformQueryEngine(
    query_engine=query_engine, query_transform=hyde
)

step_decompose_transform_gpt3 = get_step_decompose_query_transform(verbose=True)

INDEX_SUMMARY = "An index with information about Emma Stone and Ryan Gosling"

multi_step_query_engine = MultiStepQueryEngine(
    query_engine=hyde_query_engine,
    query_transform=step_decompose_transform_gpt3,
    index_summary=INDEX_SUMMARY,
)

QUERY = "Compare the families of Emma Stone and Ryan Gosling"
response = multi_step_query_engine.query(QUERY)
print(response)
