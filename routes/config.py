"""
Written by Juan Pablo Gutierrez
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/config")
def get_config():
    """
    Get the config.
    """
    