"""
GraphShield Flagship Temporal Graph Attention Network (TGAT + GRU Memory).
Dual-stage Spatio-Temporal Graph Neural Network for Cascading Vulnerability Propagation.
Combines edge-attributed directed graph attention with recurrent GRU temporal state tracking and Focal Loss.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

from graphshield.graph.temporal_dag import GraphSnapshot


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for extreme class imbalance in vulnerability cascades:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * ((1 - p_t) ** self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SpatioTemporalCell(nn.Module):
    """
    Single snapshot spatio-temporal cell:
    1. Directed GATv2 attention across dependency edges using edge attributes (constraints + age).
    2. Residual linear skip projection to retain node feature identity on isolated/sparse nodes.
    3. Gated Recurrent Unit (GRU) to update package latent memory states over epochs.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        edge_dim: int = 5,
        heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError(f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})")
        self.skip = nn.Linear(in_channels, hidden_channels)
        self.spatial_gat = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
            add_self_loops=True,
        )
        self.temporal_gru = nn.GRUCell(input_size=hidden_channels, hidden_size=hidden_channels)
        self.layer_norm = nn.LayerNorm(hidden_channels)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Residual projection preserving node attributes
        z_skip = self.skip(x)

        # Spatial attention message passing along cascade edges
        if edge_index.shape[1] > 0:
            z_gat = self.spatial_gat(x, edge_index, edge_attr=edge_attr)
            z = z_gat + z_skip
        else:
            z = z_skip

        z = F.elu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)

        # Temporal GRU state update
        if h_prev is None:
            h_next = self.temporal_gru(z)
        else:
            h_next = self.temporal_gru(z, h_prev)

        h_next = self.layer_norm(h_next)
        return h_next


class TemporalGraphShieldNet(nn.Module):
    """Full GraphShield Temporal Graph Neural Network."""

    def __init__(
        self,
        in_channels: int = 11,
        hidden_channels: int = 64,
        edge_dim: int = 5,
        heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cell = SpatioTemporalCell(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            edge_dim=edge_dim,
            heads=heads,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, snapshot_datas: List[Data]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Processes an ordered sequence of snapshot graphs {G_0, ..., G_T}.
        Returns:
            all_logits: list of logits per epoch [N, 1]
            final_hidden: final hidden representation [N, hidden_channels]
        """
        h = None
        all_logits = []
        for data in snapshot_datas:
            h = self.cell(data.x, data.edge_index, data.edge_attr, h_prev=h)
            logits = self.classifier(h).squeeze(-1)
            all_logits.append(logits)

        return all_logits, h


class GraphShieldTemporalClassifier:
    """End-to-end trainer and inference harness for GraphShield."""

    def __init__(
        self,
        in_channels: int = 11,
        hidden_channels: int = 64,
        edge_dim: int = 5,
        heads: int = 2,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        epochs: int = 70,
        alpha: float = 0.80,
        gamma: float = 2.0,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.lr = lr
        self.epochs = epochs
        self.model = TemporalGraphShieldNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            edge_dim=edge_dim,
            heads=heads,
        ).to(self.device)
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma)

    def fit(
        self,
        snapshots: List[GraphSnapshot],
        train_mask: Optional[torch.Tensor] = None,
    ) -> "GraphShieldTemporalClassifier":
        if not snapshots or snapshots[0].pyg_data.num_nodes == 0:
            return self

        mask_device = train_mask.to(self.device) if train_mask is not None else None
        if mask_device is not None and mask_device.sum() == 0:
            return self

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        # Move snapshot PyG datas to target device
        seq_datas = [s.pyg_data.to(self.device) for s in snapshots]

        for _ in range(self.epochs):
            optimizer.zero_grad()
            all_logits, _ = self.model(seq_datas)

            total_loss = torch.tensor(0.0, device=self.device)
            for t_idx, logits in enumerate(all_logits):
                targets = seq_datas[t_idx].y.to(self.device)
                if mask_device is not None:
                    total_loss = total_loss + self.criterion(logits[mask_device], targets[mask_device])
                else:
                    total_loss = total_loss + self.criterion(logits, targets)

            total_loss.backward()
            optimizer.step()

        return self

    def predict_proba(self, snapshots: List[GraphSnapshot], epoch_idx: int = -1) -> np.ndarray:
        """Predict probabilities for a given epoch snapshot."""
        if not snapshots or snapshots[0].pyg_data.num_nodes == 0:
            return np.empty((0, 2), dtype=float)
        self.model.eval()
        seq_datas = [s.pyg_data.to(self.device) for s in snapshots]

        with torch.no_grad():
            all_logits, _ = self.model(seq_datas)
            target_logits = all_logits[epoch_idx]
            p1 = torch.sigmoid(target_logits).cpu().numpy()
            p0 = 1.0 - p1

        return np.column_stack([p0, p1])

    def predict(self, snapshots: List[GraphSnapshot], epoch_idx: int = -1, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(snapshots, epoch_idx=epoch_idx)[:, 1]
        return (prob >= threshold).astype(int)
