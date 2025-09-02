"""
Written by Juan Pablo Gutierrez
"""

import os

from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
