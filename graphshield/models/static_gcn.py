"""
Baseline Static Graph Convolutional Network (GCN).
Flattens the temporal dimension into an aggregated static graph and propagates messages across dependency edges.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class StaticGCNNet(nn.Module):
    """2-Layer Graph Convolutional Network with residual skip connection."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, dropout: float = 0.2):
        super().__init__()
        self.skip = nn.Linear(in_channels, hidden_channels)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # First GCN conv + skip + ReLU
        h = self.conv1(x, edge_index) + self.skip(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Second GCN conv + ReLU
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Output logits
        out = self.classifier(h)
        return out


class StaticGCNClassifier:
    """Wrapper for training and inference with StaticGCNNet."""

    def __init__(
        self,
        in_channels: int = 11,
        hidden_channels: int = 64,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        epochs: int = 60,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.model = StaticGCNNet(in_channels, hidden_channels, out_channels=2).to(self.device)

    def fit(
        self,
        data: Data,
        train_mask: Optional[torch.Tensor] = None,
        pos_weight: Optional[float] = None,
    ) -> "StaticGCNClassifier":
        if data.num_nodes == 0 or data.x.shape[0] == 0:
            return self

        mask_device = train_mask.to(self.device) if train_mask is not None else None
        if mask_device is not None and mask_device.sum() == 0:
            return self

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = data.y.to(self.device)

        if pos_weight is None:
            y_sub = y[mask_device] if mask_device is not None else y
            n_pos = torch.sum(y_sub == 1).item()
            n_neg = torch.sum(y_sub == 0).item()
            weight_val = float(n_neg / max(n_pos, 1))
        else:
            weight_val = pos_weight

        weight_tensor = torch.tensor([1.0, weight_val], dtype=torch.float32, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self.model(x, edge_index)
            if mask_device is not None:
                loss = criterion(logits[mask_device], y[mask_device])
            else:
                loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        return self

    def predict_proba(self, data: Data) -> np.ndarray:
        if data.num_nodes == 0 or data.x.shape[0] == 0:
            return np.empty((0, 2), dtype=float)
        self.model.eval()
        with torch.no_grad():
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            logits = self.model(x, edge_index)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, data: Data, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(data)[:, 1]
        return (probs >= threshold).astype(int)
