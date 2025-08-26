from dotenv import load_dotenv

from pinecone import Pinecone
import os

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
indexes_list = map(lambda x: (x.name, x.host), pc.list_indexes())

def get_index(index_name: str):
    for index in indexes_list:
        if index[0] == index_name:
            return pc.Index(index[1])
    return None
