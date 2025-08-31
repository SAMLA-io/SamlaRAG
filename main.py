"""
Written by Juan Pablo Gutierrez
"""


# Import packages
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.llms.openai import OpenAI
from llama_index.readers.wikipedia import WikipediaReader
from dotenv import load_dotenv
import os
from vector_database.document import upsert_document
from vector_database.index import create_index, list_indexes
from pinecone.db_control.models import ServerlessSpec
from pinecone.db_control.enums import Metric, VectorType
from vector_database import pc
from vector_database.document import get_pipeline
from llama_index.vector_stores.pinecone import PineconeVectorStore
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

loader = WikipediaReader()
pages = ['Emma_Stone', 'La_La_Land', 'Ryan_Gosling']
documents = loader.load_data(pages=pages, auto_suggest=False, redirect = False)

gpt3 = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct", api_key=OPENAI_API_KEY)

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

upsert_document(index_name=index_name, namespace="test", document=documents)

def main():
    print("Hello from fastrag!")

if __name__ == "__main__":
    main()
