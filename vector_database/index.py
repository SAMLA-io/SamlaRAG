""" 
Written by Juan Pablo Gutierrez
"""

from pinecone import IndexList
from vector_database import pc
from pinecone.db_control.models import ServerlessSpec, IndexModel
from pinecone.db_control.enums import CloudProvider, AwsRegion, Metric, VectorType
from pinecone.inference import EmbedModel
from typing import Dict, Optional

def create_index_integrated_embeddings(index_name: str, cloud: CloudProvider, region: AwsRegion, model: EmbedModel, metric: Metric, field_map: Dict[str, str]) -> Optional[IndexModel]:  
    """
    Create a index with integrated embeddings.

    The index type (dense or sparse) is determined by the model.
    """
    if not pc.has_index(index_name):
        res = pc.create_index_for_model(
            name=index_name,
            cloud=cloud,
            region=region,
            metric=metric,
            embed={
                "model": model,
                "field_map": field_map,
            },
        )
        return res
    
    return None


def create_index(index_name: str, dimension: int, metric: Metric, vector_type: VectorType, spec: ServerlessSpec) -> Optional[IndexModel]:
    """
    Create an index from a backup.
    """
    if not pc.has_index(index_name):
        res = pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            vector_type=vector_type,
            spec=spec,
        )
        return res
    
    return None

def list_indexes() -> IndexList:
    """
    List all indexes.
    """
    return pc.list_indexes()
