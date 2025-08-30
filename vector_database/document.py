""" 
Written by Juan Pablo Gutierrez
"""

from vector_database import pc, get_index_host
from typing import List

def upsert_document_integrated_embeddings(index_name: str, namespace: str, document: List[dict]):
    """
    Upsert a document into a Pinecone index for integrated embedding processing.  
    """
    index = pc.Index(get_index_host(index_name))

    index.upsert_records(
        namespace=namespace,
        records=document
    )  