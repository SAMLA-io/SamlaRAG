""" 
Written by Juan Pablo Gutierrez
"""

from vector_database import pc
from pinecone.db_control.models import ServerlessSpec
from pinecone.db_control.enums import CloudProvider, AwsRegion, Metric, VectorType
from pinecone.inference import EmbedModel
from typing import Dict

"""
Create a index with integrated embeddings.

The index type (dense or sparse) is determined by the model.
"""
def create_index_integrated_embeddings(index_name: str, cloud: CloudProvider, region: AwsRegion, model: EmbedModel, metric: Metric, field_map: Dict[str, str]):  
    """
    Create a dense index with Pinecone.
    """
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud=cloud,
            region=region,
            metric=metric,
            embed={
                "model": model,
                "field_map": field_map,
            },
        )

def create_index(index_name: str, dimension: int, metric: Metric, vector_type: VectorType, spec: ServerlessSpec):
    """
    Create an index from a backup.
    """
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            vector_type=vector_type,
            spec=spec,
        )