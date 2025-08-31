""" 
Written by Juan Pablo Gutierrez
"""

from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.ingestion import IngestionPipeline
from vector_database import pc
from embeddings import get_pipeline, embed_model
from llama_index.core.schema import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from typing import List, Union
import uuid

def upsert_document(index_name: str, namespace: str, document: List[Union[dict, Document]]):
    """
    Upsert a document into a Pinecone index. This will use the ingestion pipeline to split the document 
    into nodes and embed them and store them in the vector store.
    """
    index = pc.Index(index_name)

    # Convert documents to Document objects if they're dictionaries
    documents = []
    for doc in document:
        if isinstance(doc, dict):
            documents.append(Document(text=doc["text"], metadata=doc["metadata"]))
        elif isinstance(doc, Document):
            documents.append(doc)
        else:
            raise ValueError(f"Unsupported document type: {type(doc)}")

    pipeline = get_pipeline(PineconeVectorStore(pinecone_index=index))
    pipeline.run(documents=documents, show_progress=True, insert_into_vector_store=True)