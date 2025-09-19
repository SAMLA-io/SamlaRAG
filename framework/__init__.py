"""
Declarative RAG Framework

A framework for building RAG pipelines from configuration files.
Supports declarative configuration without Python imports.
"""

from .config_reader import ConfigReader

__all__ = ["ConfigReader"]
