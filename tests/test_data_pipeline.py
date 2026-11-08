"""Unit tests for GraphShield data ingestion and out-of-core Polars/Parquet pipeline."""

from pathlib import Path
import tempfile
import pytest
import polars as pl

from graphshield.data.dataset_loader import load_ecosystem_stream, EcosystemDataset
from graphshield.data.osv_collector import OSVCollector, VulnerabilityRecord
from graphshield.data.parquet_pipeline import ParquetColumnarPipeline


@pytest.fixture
def pypi_dataset():
    return load_ecosystem_stream("pypi")


@pytest.fixture
def npm_dataset():
    return load_ecosystem_stream("npm")


def test_load_pypi_stream(pypi_dataset):
    assert pypi_dataset.ecosystem == "PyPI"
    assert pypi_dataset.num_epochs >= 5
    assert pypi_dataset.total_packages >= 40
    snap0 = pypi_dataset.get_snapshot(0)
    assert len(snap0.packages) == pypi_dataset.total_packages
    assert len(snap0.dependencies) > 0


def test_load_npm_stream(npm_dataset):
    assert npm_dataset.ecosystem == "npm"
    assert npm_dataset.num_epochs >= 5
    assert npm_dataset.total_packages >= 40
    snap0 = npm_dataset.get_snapshot(0)
    assert len(snap0.packages) == npm_dataset.total_packages
    assert len(snap0.dependencies) > 0


def test_polars_node_and_edge_conversion(pypi_dataset):
    snap = pypi_dataset.get_snapshot(1)
    node_df = snap.to_node_df()
    assert isinstance(node_df, pl.DataFrame)
    assert "maintainer_count" in node_df.columns
    assert "is_vulnerable" in node_df.columns
    assert node_df.schema["is_vulnerable"] == pl.Int32

    edge_df = snap.to_edge_df()
    assert isinstance(edge_df, pl.DataFrame)
    assert "source" in edge_df.columns
    assert "constraint_type" in edge_df.columns
    assert len(edge_df) > 0


def test_osv_collector(pypi_dataset):
    collector = OSVCollector()
    n_vulns = collector.extract_from_stream(pypi_dataset)
    assert n_vulns > 0
    assert len(collector.records) == n_vulns

    mean_lag = collector.get_mean_advisory_lag()
    assert mean_lag > 0.0

    df = collector.to_polars()
    assert isinstance(df, pl.DataFrame)
    assert "vuln_id" in df.columns
    assert "disclosure_lag_days" in df.columns
    assert len(df) == n_vulns

    # Test low severity level parsing
    low_rec = VulnerabilityRecord.from_osv_entry({
        "id": "GHSA-test-low",
        "affected": [{"package": {"name": "test-pkg", "ecosystem": "PyPI"}}],
        "database_specific": {"cvss": {"score": 3.1}}
    })
    assert low_rec.severity_level == "LOW"

    # Test numeric string in severity list
    num_rec = VulnerabilityRecord.from_osv_entry({
        "id": "GHSA-test-num",
        "affected": [{"package": {"name": "test-pkg", "ecosystem": "PyPI"}}],
        "severity": [{"type": "CVSS_V3", "score": "8.8"}]
    })
    assert num_rec.severity_score == 8.8
    assert num_rec.severity_level == "HIGH"


def test_parquet_columnar_pipeline(pypi_dataset):
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = ParquetColumnarPipeline(storage_dir=tmp_dir)
        summary = pipeline.ingest_dataset(pypi_dataset)

        assert summary["node_count"] > 0
        assert summary["edge_count"] > 0
        assert summary["epochs"] == pypi_dataset.num_epochs

        # Test querying snapshot with Polars lazy scan
        snap2_nodes = pipeline.query_snapshot_nodes(epoch=2)
        assert len(snap2_nodes) == pypi_dataset.total_packages

        snap2_edges = pipeline.query_snapshot_edges(epoch=2)
        assert len(snap2_edges) > 0

        # Test DuckDB analytical degree distribution join
        deg_df = pipeline.query_temporal_degree_distribution()
        assert isinstance(deg_df, pl.DataFrame)
        assert "in_degree" in deg_df.columns
        assert "out_degree" in deg_df.columns

        # Test high-level summary stats
        stats = pipeline.compute_summary_stats()
        assert stats["unique_packages"] == pypi_dataset.total_packages
        assert stats["num_epochs"] == pypi_dataset.num_epochs


def test_parquet_pipeline_uninitialized_guard():
    """Verify that querying an un-ingested pipeline raises RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = ParquetColumnarPipeline(storage_dir=tmp_dir)
        with pytest.raises(RuntimeError, match="Parquet tables not found"):
            pipeline.compute_summary_stats()
        with pytest.raises(RuntimeError, match="Parquet tables not found"):
            pipeline.query_temporal_degree_distribution()
