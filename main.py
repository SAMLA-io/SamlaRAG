"""
Written by Juan Pablo Gutierrez
"""


from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from vector_database.index import create_index, list_indexes
from pinecone.db_control.enums import Metric, VectorType
from pinecone.db_control.models import ServerlessSpec
from llama_index.core import VectorStoreIndex
from vector_database import pc
from dotenv import load_dotenv

load_dotenv()

print("Existing indexes:")
indexes = list_indexes()
for index in indexes:
    print(f"  - {index.name}")

index_name = "rag"
if not any(index.name == index_name for index in indexes):
    print(f"Creating index '{index_name}'...")
    create_index(
        index_name=index_name, 
        dimension=1536, 
        metric=Metric.COSINE, 
        vector_type=VectorType.DENSE, 
        spec=ServerlessSpec(region="us-east-1", cloud="aws")
    )
    print(f"Index '{index_name}' created successfully!")
else:
    print(f"Index '{index_name}' already exists.")

vector_store = PineconeVectorStore(pinecone_index=pc.Index(index_name))
vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# This is the most important part of the RAG
retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)
