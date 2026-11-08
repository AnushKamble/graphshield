"""Unit tests for Dynamic Temporal DAG and Topological Feature Extractor."""

import networkx as nx
import numpy as np
import pytest
import torch

from graphshield.data.dataset_loader import load_ecosystem_stream
from graphshield.graph.temporal_dag import TemporalDAG, GraphSnapshot
from graphshield.graph.topology_features import (
    extract_topological_features,
    compute_downstream_blast_radius,
    compute_transitive_depth,
)


@pytest.fixture
def pypi_dag():
    dataset = load_ecosystem_stream("pypi")
    return TemporalDAG(dataset)


def test_temporal_dag_snapshots(pypi_dag):
    assert len(pypi_dag.snapshots) == 5
    snap0 = pypi_dag.get_snapshot(0)
    assert isinstance(snap0, GraphSnapshot)
    assert snap0.epoch == 0
    assert len(snap0.node_to_idx) == len(snap0.idx_to_node)

    # PyG tensor verification
    pyg_data = snap0.pyg_data
    assert pyg_data.x.shape[0] == len(snap0.node_to_idx)
    assert pyg_data.x.shape[1] == 11  # 11 node features
    assert pyg_data.edge_index.shape[0] == 2
    assert pyg_data.edge_attr.shape[1] == 5  # 4 one-hot + 1 age
    assert pyg_data.y.shape[0] == len(snap0.node_to_idx)

    # Test standardization with subset of snapshots (prevent leakage)
    m, s = pypi_dag.standardize_features(pypi_dag.snapshots[:3])
    assert m.shape == (1, 11)
    assert s.shape == (1, 11)


def test_topological_feature_extraction():
    # Build a small known DAG: C -> B -> A (C depends on B, B depends on A)
    g = nx.DiGraph()
    g.add_edge("B", "A", constraint="caret", age=100)
    g.add_edge("C", "B", constraint="pinned", age=50)

    # If A is compromised, both B and C depend on A, so blast radius of A is 2 (B, C)
    blast_a = compute_downstream_blast_radius(g, "A")
    assert blast_a == 2

    blast_b = compute_downstream_blast_radius(g, "B")
    assert blast_b == 1

    blast_c = compute_downstream_blast_radius(g, "C")
    assert blast_c == 0

    features = extract_topological_features(g, ["A", "B", "C"])
    assert "A" in features
    assert features["A"]["in_degree"] == 1.0  # B -> A
    assert features["C"]["out_degree"] == 1.0  # C -> B
    assert features["C"]["unpinned_ratio"] == 0.0  # pinned


def test_static_aggregated_graph(pypi_dag):
    static_data = pypi_dag.get_static_aggregated_graph()
    assert static_data.x.shape[0] == len(pypi_dag.snapshots[0].node_to_idx)
    assert static_data.edge_index.shape[0] == 2
    assert static_data.edge_index.shape[1] > 0
    assert static_data.y.shape[0] == static_data.x.shape[0]


def test_tabular_dataset_export(pypi_dag):
    X, y, feature_names = pypi_dag.get_tabular_dataset()
    expected_rows = len(pypi_dag.snapshots) * len(pypi_dag.snapshots[0].node_to_idx)
    assert X.shape[0] == expected_rows
    assert len(y) == expected_rows
    assert len(feature_names) == 11
    assert "blast_radius" in feature_names
    assert "pagerank" in feature_names


def test_edge_index_cascade_direction(pypi_dag):
    """
    Verify that in PyG message passing, edges flow from upstream dependency
    to downstream consumer (v -> u) along the vulnerability cascade path.
    """
    snap = pypi_dag.snapshots[0]
    u_idx = snap.node_to_idx["requests"]
    v_idx = snap.node_to_idx["urllib3"]

    edge_index = snap.pyg_data.edge_index
    # Finding if there is an edge from urllib3 (v_idx) to requests (u_idx)
    found_cascade_edge = False
    for i in range(edge_index.shape[1]):
        if edge_index[0, i].item() == v_idx and edge_index[1, i].item() == u_idx:
            found_cascade_edge = True
            break
    assert found_cascade_edge, "Expected PyG edge_index to direct message passing from urllib3 (source of vulnerability) to requests (consumer)."


def test_topological_features_with_cycles_and_isolated_nodes():
    """Verify that circular dependencies and isolated packages don't crash feature extraction."""
    g = nx.DiGraph()
    # Cycle: A -> B -> A
    g.add_edge("A", "B", constraint="caret")
    g.add_edge("B", "A", constraint="tilde")
    # Isolated node
    g.add_node("C")

    features = extract_topological_features(g, ["A", "B", "C"])
    assert "A" in features
    assert "B" in features
    assert "C" in features
    assert features["C"]["in_degree"] == 0.0
    assert features["C"]["out_degree"] == 0.0
    assert features["C"]["blast_radius"] == 0.0


def test_empty_dataset_temporal_dag():
    """Verify that TemporalDAG handles datasets with empty snapshots cleanly."""
    from graphshield.data.dataset_loader import EcosystemDataset, SnapshotData
    empty_ds = EcosystemDataset(
        ecosystem="EmptyEco",
        description="Empty",
        num_epochs=1,
        total_packages=0,
        snapshots=[SnapshotData(epoch=0, timestamp="2023-01-01", packages=[], dependencies=[])],
    )
    empty_dag = TemporalDAG(empty_ds)
    assert len(empty_dag.snapshots) == 1
    snap = empty_dag.get_snapshot(0)
    assert snap.raw_node_features.shape == (0, 11)
    assert snap.labels.shape == (0,)
    assert snap.pyg_data.x.shape == (0, 11)
    assert snap.pyg_data.edge_index.shape == (2, 0)

    # Static aggregated graph
    static_agg = empty_dag.get_static_aggregated_graph()
    assert static_agg.num_nodes == 0
    assert static_agg.edge_index.shape == (2, 0)

    # Tabular dataset
    X, y, _ = empty_dag.get_tabular_dataset()
    assert X.shape == (0, 11)
    assert y.shape == (0,)
