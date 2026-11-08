"""
Dynamic Multi-Snapshot Directed Acyclic Dependency Graph (DAG) Representation.
Builds temporal graph sequences {G_0, ..., G_T} with node and edge attributes for PyTorch Geometric.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from graphshield.data.dataset_loader import EcosystemDataset, SnapshotData
from graphshield.graph.topology_features import extract_topological_features


CONSTRAINT_MAP = {
    "pinned": 0,
    "tilde": 1,
    "caret": 2,
    "wildcard": 3,
}


@dataclass
class GraphSnapshot:
    """A single temporal snapshot G_t = (V_t, E_t, X_t, E_t_attr, y_t)."""
    epoch: int
    timestamp: str
    node_to_idx: Dict[str, int]
    idx_to_node: Dict[int, str]
    nx_graph: nx.DiGraph
    pyg_data: Data
    raw_node_features: np.ndarray
    labels: np.ndarray
    feature_names: List[str]


class TemporalDAG:
    """Dynamic multi-epoch sequence of directed dependency graph snapshots."""

    def __init__(self, dataset: EcosystemDataset):
        self.dataset = dataset
        self.ecosystem = dataset.ecosystem
        self.snapshots: List[GraphSnapshot] = []
        self._build_snapshots()

    def _build_snapshots(self) -> None:
        """Construct GraphSnapshot for each epoch in the dataset."""
        # Fix a canonical package universe across snapshots to ensure consistent node ordering
        all_pkg_names = sorted(list({
            pkg["name"]
            for s in self.dataset.snapshots
            for pkg in s.packages
        }))
        canonical_map = {name: idx for idx, name in enumerate(all_pkg_names)}
        canonical_rev = {idx: name for idx, name in enumerate(all_pkg_names)}
        n_nodes = len(all_pkg_names)

        # We first build the topological graphs to augment node features
        for snapshot in self.dataset.snapshots:
            epoch = snapshot.epoch
            timestamp = snapshot.timestamp

            # Directed graph where edge (u, v) means u depends on v
            # Note: in vulnerability propagation, infection flows v -> u (upstream to downstream)
            g = nx.DiGraph()
            for name in all_pkg_names:
                g.add_node(name)

            edge_sources = []
            edge_targets = []
            edge_attrs = []

            for dep in snapshot.dependencies:
                src = dep["source"]
                tgt = dep["target"]
                ctype = dep.get("constraint_type", "caret").lower()
                age = float(dep.get("age_days", 0.0))

                if src in canonical_map and tgt in canonical_map:
                    g.add_edge(src, tgt, constraint=ctype, age=age)
                    u = canonical_map[src]
                    v = canonical_map[tgt]
                    edge_sources.append(u)
                    edge_targets.append(v)

                    # Edge feature: 4-dim one-hot for constraint + 1-dim normalized age
                    one_hot = [0.0] * 4
                    c_idx = CONSTRAINT_MAP.get(ctype, 2)
                    one_hot[c_idx] = 1.0
                    norm_age = min(age / 1000.0, 5.0)
                    edge_attrs.append(one_hot + [norm_age])

            # Extract topological metrics for this snapshot
            topo_metrics = extract_topological_features(g, all_pkg_names)

            # Build node feature vectors X_t
            # Map packages in snapshot
            pkg_data_map = {p["name"]: p for p in snapshot.packages}
            node_features = []
            labels = []

            feature_names = [
                "maintainer_count",
                "commit_cadence",
                "release_velocity",
                "historical_cve_count",
                "in_degree",
                "out_degree",
                "pagerank",
                "betweenness",
                "k_core",
                "blast_radius",
                "unpinned_ratio",
            ]

            for name in all_pkg_names:
                p_info = pkg_data_map.get(name, {})
                maint = float(p_info.get("maintainer_count", 1.0))
                commit = float(p_info.get("commit_cadence", 1.0))
                rel = float(p_info.get("release_velocity", 1.0))
                hist_cve = float(p_info.get("historical_cve_count", 0.0))
                lbl = int(p_info.get("is_vulnerable", 0))

                t_feat = topo_metrics[name]
                feat_vec = [
                    maint,
                    commit,
                    rel,
                    hist_cve,
                    t_feat["in_degree"],
                    t_feat["out_degree"],
                    t_feat["pagerank"] * 100.0,  # scaled for numerical stability
                    t_feat["betweenness"] * 100.0,
                    t_feat["k_core"],
                    t_feat["blast_radius"],
                    t_feat["unpinned_ratio"],
                ]
                node_features.append(feat_vec)
                labels.append(lbl)

            if node_features:
                x_arr = np.array(node_features, dtype=np.float32)
                y_arr = np.array(labels, dtype=np.int64)
            else:
                x_arr = np.empty((0, len(feature_names)), dtype=np.float32)
                y_arr = np.empty((0,), dtype=np.int64)

            # In vulnerability cascades, compromise propagates from upstream dependency (v)
            # downstream to dependent consumer (u). In PyG, messages pass edge_index[0] -> edge_index[1].
            # Therefore, we direct message passing along the cascade propagation direction:
            # edge_index[0] = upstream dependency (edge_targets), edge_index[1] = consumer (edge_sources).
            if edge_sources:
                edge_index = torch.tensor([edge_targets, edge_sources], dtype=torch.long)
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 5), dtype=torch.float32)

            pyg_data = Data(
                x=torch.from_numpy(x_arr),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.from_numpy(y_arr),
                num_nodes=n_nodes,
            )

            snap_obj = GraphSnapshot(
                epoch=epoch,
                timestamp=timestamp,
                node_to_idx=canonical_map,
                idx_to_node=canonical_rev,
                nx_graph=g,
                pyg_data=pyg_data,
                raw_node_features=x_arr,
                labels=y_arr,
                feature_names=feature_names,
            )
            self.snapshots.append(snap_obj)

        # Standardize node features
        self.standardize_features()

    def standardize_features(
        self,
        fit_snapshots: Optional[List[GraphSnapshot]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardize raw node features across snapshots to prevent temporal data leakage.
        If fit_snapshots is provided, mean and std are computed ONLY on those snapshots
        (e.g., training epochs) and applied across all snapshots.
        """
        ref_snapshots = fit_snapshots if fit_snapshots is not None else self.snapshots
        if not ref_snapshots or ref_snapshots[0].raw_node_features.shape[0] == 0:
            feat_mean = np.zeros((1, 11), dtype=np.float32)
            feat_std = np.ones((1, 11), dtype=np.float32)
            return feat_mean, feat_std
        all_x_mats = np.vstack([s.raw_node_features for s in ref_snapshots])
        feat_mean = np.mean(all_x_mats, axis=0, keepdims=True).astype(np.float32)
        feat_std = (np.std(all_x_mats, axis=0, keepdims=True) + 1e-6).astype(np.float32)
        for s in self.snapshots:
            if s.raw_node_features.shape[0] > 0:
                norm_x = (s.raw_node_features - feat_mean) / feat_std
                s.pyg_data.x = torch.from_numpy(norm_x)
        return feat_mean, feat_std

    def get_snapshot(self, epoch: int) -> GraphSnapshot:
        for s in self.snapshots:
            if s.epoch == epoch:
                return s
        raise IndexError(f"Epoch {epoch} not found.")

    def get_tabular_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Return flattened tabular (X, y, feature_names) for classical ML baselines."""
        all_X = []
        all_y = []
        for s in self.snapshots:
            all_X.append(s.raw_node_features)
            all_y.append(s.labels)
        if not all_X or all_X[0].shape[0] == 0:
            feat_names = self.snapshots[0].feature_names if self.snapshots else []
            return np.empty((0, len(feat_names)), dtype=np.float32), np.empty((0,), dtype=np.int64), feat_names
        X_mat = np.vstack(all_X)
        y_vec = np.concatenate(all_y)
        return X_mat, y_vec, self.snapshots[0].feature_names

    def get_static_aggregated_graph(self) -> Data:
        """
        Aggregate temporal snapshots into a single static graph Data object
        for the Static GCN baseline.
        """
        if not self.snapshots or self.snapshots[0].raw_node_features.shape[0] == 0:
            return Data(
                x=torch.empty((0, 11), dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                y=torch.empty((0,), dtype=torch.long),
                num_nodes=0,
            )
        # Average standardized node features across all epochs
        all_x = np.stack([s.pyg_data.x.detach().cpu().numpy() for s in self.snapshots], axis=0)
        mean_x = np.mean(all_x, axis=0)

        # Union of all edges across epochs
        edge_set = set()
        for s in self.snapshots:
            edge_idx = s.pyg_data.edge_index.detach().cpu().numpy()
            for i in range(edge_idx.shape[1]):
                edge_set.add((edge_idx[0, i], edge_idx[1, i]))

        if edge_set:
            srcs = [e[0] for e in edge_set]
            tgts = [e[1] for e in edge_set]
            edge_index = torch.tensor([srcs, tgts], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Label is 1 if node was vulnerable in any epoch
        labels = np.max(np.stack([s.labels for s in self.snapshots], axis=0), axis=0)

        return Data(
            x=torch.from_numpy(mean_x),
            edge_index=edge_index,
            y=torch.from_numpy(labels),
            num_nodes=len(mean_x),
        )
