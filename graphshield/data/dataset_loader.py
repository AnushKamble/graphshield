"""
Ecosystem Snapshot Loader for GraphShield.
Loads dynamic multi-epoch dependency streams for PyPI and npm.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import polars as pl


@dataclass
class SnapshotData:
    epoch: int
    timestamp: str
    packages: List[Dict[str, Any]]
    dependencies: List[Dict[str, Any]]

    def to_node_df(self) -> pl.DataFrame:
        """Convert snapshot packages to a typed Polars DataFrame."""
        rows = []
        for pkg in self.packages:
            rows.append({
                "epoch": self.epoch,
                "timestamp": self.timestamp,
                "name": str(pkg["name"]),
                "version": str(pkg.get("version", "1.0.0")),
                "maintainer_count": int(pkg.get("maintainer_count", 1)),
                "commit_cadence": float(pkg.get("commit_cadence", 1.0)),
                "release_velocity": float(pkg.get("release_velocity", 1.0)),
                "historical_cve_count": int(pkg.get("historical_cve_count", 0)),
                "is_vulnerable": int(pkg.get("is_vulnerable", 0)),
                "vuln_count": len(pkg.get("vulnerability_details", [])),
            })
        return pl.DataFrame(rows, schema={
            "epoch": pl.Int32,
            "timestamp": pl.Utf8,
            "name": pl.Utf8,
            "version": pl.Utf8,
            "maintainer_count": pl.Int32,
            "commit_cadence": pl.Float64,
            "release_velocity": pl.Float64,
            "historical_cve_count": pl.Int32,
            "is_vulnerable": pl.Int32,
            "vuln_count": pl.Int32,
        })

    def to_edge_df(self) -> pl.DataFrame:
        """Convert snapshot dependencies to a typed Polars DataFrame."""
        rows = []
        for dep in self.dependencies:
            rows.append({
                "epoch": self.epoch,
                "source": str(dep["source"]),
                "target": str(dep["target"]),
                "constraint_type": str(dep.get("constraint_type", "caret")),
                "version_spec": str(dep.get("version_spec", "*")),
                "age_days": int(dep.get("age_days", 0)),
            })
        return pl.DataFrame(rows, schema={
            "epoch": pl.Int32,
            "source": pl.Utf8,
            "target": pl.Utf8,
            "constraint_type": pl.Utf8,
            "version_spec": pl.Utf8,
            "age_days": pl.Int32,
        })


@dataclass
class EcosystemDataset:
    ecosystem: str
    description: str
    num_epochs: int
    total_packages: int
    snapshots: List[SnapshotData]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_snapshot(self, epoch: int) -> SnapshotData:
        for s in self.snapshots:
            if s.epoch == epoch:
                return s
        raise IndexError(f"Epoch {epoch} not found. Available epochs: {[s.epoch for s in self.snapshots]}")

    def all_nodes_df(self) -> pl.DataFrame:
        """Return combined Polars DataFrame for all nodes across epochs."""
        dfs = [s.to_node_df() for s in self.snapshots]
        return pl.concat(dfs) if dfs else pl.DataFrame()

    def all_edges_df(self) -> pl.DataFrame:
        """Return combined Polars DataFrame for all edges across epochs."""
        dfs = [s.to_edge_df() for s in self.snapshots]
        return pl.concat(dfs) if dfs else pl.DataFrame()


def load_ecosystem_stream(target: Union[str, Path, Dict[str, Any]]) -> EcosystemDataset:
    """
    Load an ecosystem dependency stream from file path, ecosystem identifier ('pypi', 'npm'),
    or in-memory dictionary.
    """
    data: Dict[str, Any]
    if isinstance(target, dict):
        data = target
    else:
        path_str = str(target)
        # Check if short alias
        if path_str.lower() in ("pypi", "pypi_dependency_stream"):
            base_dir = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "datasets"
            candidate = base_dir / "pypi_dependency_stream.json"
            if candidate.exists():
                path_str = str(candidate)
        elif path_str.lower() in ("npm", "npm_dependency_stream"):
            base_dir = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "datasets"
            candidate = base_dir / "npm_dependency_stream.json"
            if candidate.exists():
                path_str = str(candidate)

        filepath = Path(path_str)
        if not filepath.exists():
            raise FileNotFoundError(f"Dependency stream dataset not found at: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

    snapshots = []
    for item in data.get("snapshots", []):
        snapshots.append(SnapshotData(
            epoch=int(item["epoch"]),
            timestamp=str(item.get("timestamp", "")),
            packages=list(item.get("packages", [])),
            dependencies=list(item.get("dependencies", [])),
        ))

    return EcosystemDataset(
        ecosystem=data.get("ecosystem", "Unknown"),
        description=data.get("description", ""),
        num_epochs=len(snapshots),
        total_packages=data.get("total_packages", len(snapshots[0].packages) if snapshots else 0),
        snapshots=snapshots,
        metadata={"source": str(target) if not isinstance(target, dict) else "memory"}
    )
