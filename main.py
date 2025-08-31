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
from llama_index.core.retrievers import VectorIndexRetriever
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

def main():
    print("Hello from fastrag!")

vector_store = PineconeVectorStore(pinecone_index=pc.Index(index_name))
vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.query_engine import RetrieverQueryEngine

# Pass in your retriever from above, which is configured to return the top 5 results
query_engine = RetrieverQueryEngine(retriever=retriever)


while True:
    # Now you query:
    llm_query = query_engine.query(input('Enter your query: '))

    print(llm_query.response)
# Response:
# 'Logarithmic complexity in graph construction affects the construction process by organizing the graph into different layers based on their length scale. This separation of links into layers allows for efficient and scalable routing in the graph. The construction algorithm starts from the top layer, which contains the longest links, and greedily traverses through the elements until a local minimum is reached. Then, the search switches to the lower layer with shorter links, and the process repeats. By keeping the maximum number of connections per element constant in all layers, the routing complexity in the graph scales logarithmically. This logarithmic complexity is achieved by assigning an integer level to each element, determining the maximum layer it belongs to. The construction algorithm incrementally builds a proximity graph for each layer, consisting of "short" links that approximate the Delaunay graph. Overall, logarithmic complexity in graph construction enables efficient and robust approximate nearest neighbor search.'
if __name__ == "__main__":
    main()
