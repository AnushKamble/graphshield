"""Unit tests for GraphShield baseline models and Temporal GNN architecture."""

import numpy as np
import pytest
import torch

from graphshield.data.dataset_loader import load_ecosystem_stream
from graphshield.graph.temporal_dag import TemporalDAG
from graphshield.models.baselines import PageRankBaseline, RandomForestBaseline, XGBoostBaseline
from graphshield.models.static_gcn import StaticGCNNet, StaticGCNClassifier
from graphshield.models.temporal_gnn import (
    TemporalGraphShieldNet,
    GraphShieldTemporalClassifier,
    FocalLoss,
)
from graphshield.models.mock_engine import DeterministicMockEngine


@pytest.fixture
def pypi_dag():
    dataset = load_ecosystem_stream("pypi")
    return TemporalDAG(dataset)


def test_pagerank_baseline(pypi_dag):
    X, y, _ = pypi_dag.get_tabular_dataset()
    model = PageRankBaseline()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (X.shape[0], 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)


def test_random_forest_baseline(pypi_dag):
    X, y, _ = pypi_dag.get_tabular_dataset()
    model = RandomForestBaseline(n_estimators=10, max_depth=3)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (X.shape[0], 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)


def test_xgboost_baseline(pypi_dag):
    X, y, _ = pypi_dag.get_tabular_dataset()
    model = XGBoostBaseline(n_estimators=10, max_depth=3)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (X.shape[0], 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)


def test_static_gcn(pypi_dag):
    static_data = pypi_dag.get_static_aggregated_graph()
    clf = StaticGCNClassifier(in_channels=11, hidden_channels=32, epochs=5, device="cpu")
    clf.fit(static_data)
    probs = clf.predict_proba(static_data)
    assert probs.shape == (static_data.num_nodes, 2)
    preds = clf.predict(static_data)
    assert preds.shape == (static_data.num_nodes,)


def test_temporal_gnn_forward_and_focal_loss(pypi_dag):
    # Test raw model forward pass and loss
    model = TemporalGraphShieldNet(in_channels=11, hidden_channels=32, edge_dim=5, heads=2)
    datas = [s.pyg_data for s in pypi_dag.snapshots]

    all_logits, final_h = model(datas)
    assert len(all_logits) == len(datas)
    assert all_logits[0].shape == (datas[0].num_nodes,)
    assert final_h.shape == (datas[0].num_nodes, 32)

    # Test FocalLoss
    focal = FocalLoss(alpha=0.75, gamma=2.0)
    loss = focal(all_logits[0], datas[0].y)
    assert loss.dim() == 0
    assert loss.item() > 0.0

    # Test backward pass
    loss.backward()
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any()


def test_temporal_gnn_classifier_pipeline(pypi_dag):
    clf = GraphShieldTemporalClassifier(
        in_channels=11,
        hidden_channels=32,
        edge_dim=5,
        heads=2,
        epochs=3,
        device="cpu",
    )
    clf.fit(pypi_dag.snapshots)
    probs = clf.predict_proba(pypi_dag.snapshots)
    assert probs.shape == (pypi_dag.snapshots[0].pyg_data.num_nodes, 2)
    preds = clf.predict(pypi_dag.snapshots)
    assert preds.shape == (pypi_dag.snapshots[0].pyg_data.num_nodes,)


def test_deterministic_mock_engine():
    engine = DeterministicMockEngine()
    pypi_res = engine.get_benchmark_results("pypi")
    assert "GraphShield (Proposed)" in pypi_res
    assert pypi_res["GraphShield (Proposed)"]["pr_auc"] > 0.85

    npm_res = engine.get_benchmark_results("npm")
    assert "GraphShield (Proposed)" in npm_res
    assert npm_res["GraphShield (Proposed)"]["roc_auc"] > 0.90

    probs = engine.generate_mock_probabilities(20)
    assert probs.shape == (20, 2)


def test_spatio_temporal_cell_isolated_nodes():
    """Verify that SpatioTemporalCell handles zero-edge graphs without zeroing out features."""
    from graphshield.models.temporal_gnn import SpatioTemporalCell
    cell = SpatioTemporalCell(in_channels=11, hidden_channels=32, edge_dim=5, heads=2)

    # 3 nodes, 0 edges
    x = torch.randn(3, 11)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 5), dtype=torch.float32)

    h = cell(x, edge_index, edge_attr, h_prev=None)
    assert h.shape == (3, 32)
    # Output must not be all zeros since node has non-zero features
    assert torch.sum(torch.abs(h)).item() > 0.01

    # Verify ValueError raised if hidden_channels not divisible by heads
    with pytest.raises(ValueError, match="must be divisible by heads"):
        SpatioTemporalCell(in_channels=11, hidden_channels=33, edge_dim=5, heads=2)


def test_train_mask_support(pypi_dag):
    """Verify that both Static GCN and Temporal GraphShield train properly with train_mask."""
    n_nodes = pypi_dag.snapshots[0].pyg_data.num_nodes
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    mask[: n_nodes // 2] = True

    # Static GCN with mask
    static_data = pypi_dag.get_static_aggregated_graph()
    gcn = StaticGCNClassifier(in_channels=11, hidden_channels=32, epochs=3, device="cpu")
    gcn.fit(static_data, train_mask=mask)
    probs_gcn = gcn.predict_proba(static_data)
    assert probs_gcn.shape == (n_nodes, 2)

    # Temporal GNN with mask
    gnn = GraphShieldTemporalClassifier(in_channels=11, hidden_channels=32, edge_dim=5, heads=2, epochs=2, device="cpu")
    gnn.fit(pypi_dag.snapshots, train_mask=mask)
    probs_gnn = gnn.predict_proba(pypi_dag.snapshots)
    assert probs_gnn.shape == (n_nodes, 2)


def test_models_robust_edge_cases(pypi_dag):
    """Verify models handle empty inputs, single-class labels, and all-false masks gracefully."""
    # 1. PageRankBaseline on empty array
    pr = PageRankBaseline()
    empty_pr = pr.predict_proba(np.empty((0, 11)))
    assert empty_pr.shape == (0, 2)

    # 2. XGBoostBaseline on single-class labels
    xgb_single = XGBoostBaseline(n_estimators=5, max_depth=2)
    xgb_single.fit(np.ones((6, 11)), np.zeros(6, dtype=int))
    probs_xgb = xgb_single.predict_proba(np.ones((3, 11)))
    assert probs_xgb.shape == (3, 2)
    assert np.allclose(probs_xgb[:, 1], 0.0)

    # 3. StaticGCNClassifier on empty graph and all-false mask
    from torch_geometric.data import Data
    empty_data = Data(x=torch.empty((0, 11)), edge_index=torch.empty((2, 0), dtype=torch.long), y=torch.empty((0,), dtype=torch.long), num_nodes=0)
    gcn = StaticGCNClassifier(in_channels=11, hidden_channels=16, epochs=1, device="cpu")
    gcn.fit(empty_data)
    assert gcn.predict_proba(empty_data).shape == (0, 2)

    # All-false mask on non-empty graph
    static_data = pypi_dag.get_static_aggregated_graph()
    false_mask = torch.zeros(static_data.num_nodes, dtype=torch.bool)
    gcn.fit(static_data, train_mask=false_mask)
    probs_false = gcn.predict_proba(static_data)
    assert not np.isnan(probs_false).any()

    # 4. GraphShieldTemporalClassifier on empty snapshots and all-false mask
    gnn = GraphShieldTemporalClassifier(in_channels=11, hidden_channels=16, edge_dim=5, heads=2, epochs=1, device="cpu")
    gnn.fit([])
    assert gnn.predict_proba([]).shape == (0, 2)

    gnn.fit(pypi_dag.snapshots, train_mask=false_mask)
    probs_gnn_false = gnn.predict_proba(pypi_dag.snapshots)
    assert not np.isnan(probs_gnn_false).any()
