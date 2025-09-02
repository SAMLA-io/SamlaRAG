"""
Written by Juan Pablo Gutierrez
"""

from pinecone.db_control.models import ServerlessSpec
from pinecone.db_control.enums import Metric, VectorType
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from dotenv import load_dotenv
from reranking.cohere import rerank
from vector_database.index import create_index, list_indexes
from vector_database import pc

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
response = retriever.retrieve("Compare the families of Emma Stone and Ryan Gosling")

documents = [node.text for node in response]

results = rerank(
    model="rerank-v3.5",
    query="Compare the families of Emma Stone and Ryan Gosling",
    documents=documents,
    top_n=5,
)

for result in results.results:
    print(documents[result.index][:500])
    print(result.relevance_score)
    print("--------------------------------")
