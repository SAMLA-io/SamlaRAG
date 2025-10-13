"""
Written by Juan Pablo Gutierrez
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.get("/")   
def read_root():
    """
    Root endpoint.
    """
    return {"message": "Welcome to SamlaRAG!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)