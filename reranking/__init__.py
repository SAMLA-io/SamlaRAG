"""
Written by Juan Pablo Gutierrez
"""

import os
import cohere

from dotenv import load_dotenv

load_dotenv()

co = cohere.Client(api_key=os.getenv("CO_API_KEY"))
