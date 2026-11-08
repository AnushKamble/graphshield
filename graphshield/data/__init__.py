"""Data ingestion and out-of-core pipeline for GraphShield."""

from graphshield.data.dataset_loader import load_ecosystem_stream, EcosystemDataset
from graphshield.data.osv_collector import OSVCollector, VulnerabilityRecord
from graphshield.data.parquet_pipeline import ParquetColumnarPipeline

__all__ = [
    "load_ecosystem_stream",
    "EcosystemDataset",
    "OSVCollector",
    "VulnerabilityRecord",
    "ParquetColumnarPipeline",
]
