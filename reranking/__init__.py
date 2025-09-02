from dotenv import load_dotenv
import cohere 
import os

load_dotenv()

co = cohere.Client(api_key=os.getenv("CO_API_KEY"))
