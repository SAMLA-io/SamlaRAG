from dotenv import load_dotenv

from pinecone import Pinecone
import os

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
