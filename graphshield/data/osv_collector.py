"""
Standardized OSV / GHSA Vulnerability Ground Truth Collector and Feeder.
Parses Open Source Vulnerability schema and tracks advisory disclosure lag and cascade relations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import polars as pl


@dataclass
class VulnerabilityRecord:
    vuln_id: str
    package_name: str
    ecosystem: str
    severity_score: float  # CVSS v3 score (0.0 to 10.0)
    severity_level: str   # LOW, MEDIUM, HIGH, CRITICAL
    disclosure_lag_days: int
    published_date: str
    introduced_date: Optional[str] = None
    is_transitive_cascade: bool = False
    parent_source_package: Optional[str] = None
    cve_aliases: List[str] = field(default_factory=list)
    raw_osv_payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_osv_entry(cls, osv_dict: Dict[str, Any], ecosystem_default: str = "PyPI") -> "VulnerabilityRecord":
        """Parse an OSV format JSON dictionary into a VulnerabilityRecord."""
        vuln_id = osv_dict.get("id", "UNKNOWN-VULN")
        aliases = osv_dict.get("aliases", [])
        
        # Extract affected package
        pkg_name = "unknown"
        eco = ecosystem_default
        affected_list = osv_dict.get("affected", [])
        if affected_list:
            pkg_info = affected_list[0].get("package", {})
            pkg_name = pkg_info.get("name", "unknown")
            eco = pkg_info.get("ecosystem", ecosystem_default)

        # Severity score
        severity_score = 7.5
        severity_level = "HIGH"
        severities = osv_dict.get("severity", [])
        if severities and isinstance(severities, list):
            score_val = severities[0].get("score", "")
            if isinstance(score_val, (int, float)):
                severity_score = float(score_val)
            elif isinstance(score_val, str):
                try:
                    severity_score = float(score_val)
                except ValueError:
                    if "CVSS:" in score_val:
                        severity_score = 7.5
        if "database_specific" in osv_dict:
            db_spec = osv_dict["database_specific"]
            if "cvss" in db_spec and isinstance(db_spec["cvss"], dict):
                severity_score = float(db_spec["cvss"].get("score", severity_score))
            severity_level = db_spec.get("severity", severity_level).upper()

        if severity_score >= 9.0:
            severity_level = "CRITICAL"
        elif severity_score >= 7.0:
            severity_level = "HIGH"
        elif severity_score >= 4.0:
            severity_level = "MEDIUM"
        else:
            severity_level = "LOW"

        pub_date = osv_dict.get("published", "2023-01-01T00:00:00Z")
        intro_date = osv_dict.get("introduced", None)
        lag_days = int(osv_dict.get("disclosure_lag_days", 60))

        is_transitive = bool(osv_dict.get("is_transitive_cascade", False) or "CASCADE" in vuln_id)
        parent_pkg = osv_dict.get("cascade_from", None)

        return cls(
            vuln_id=vuln_id,
            package_name=pkg_name,
            ecosystem=eco,
            severity_score=severity_score,
            severity_level=severity_level,
            disclosure_lag_days=lag_days,
            published_date=pub_date,
            introduced_date=intro_date,
            is_transitive_cascade=is_transitive,
            parent_source_package=parent_pkg,
            cve_aliases=aliases,
            raw_osv_payload=osv_dict,
        )


class OSVCollector:
    """Ground truth vulnerability collector for software dependency graphs."""

    def __init__(self, records: Optional[List[VulnerabilityRecord]] = None):
        self.records: List[VulnerabilityRecord] = records or []

    def add_record(self, record: VulnerabilityRecord) -> None:
        self.records.append(record)

    def extract_from_stream(self, ecosystem_stream: Any) -> int:
        """Extract all vulnerability records embedded in an EcosystemDataset."""
        count = 0
        for snapshot in ecosystem_stream.snapshots:
            for pkg in snapshot.packages:
                for vuln in pkg.get("vulnerability_details", []):
                    cve_id = vuln.get("cve") or vuln.get("cve_id") or "CVE-UNKNOWN"
                    lag = int(vuln.get("lag_days", vuln.get("disclosure_lag_days", 60)))
                    sev = float(vuln.get("severity", 7.5))
                    parent = vuln.get("cascade_from", None)
                    is_casc = bool(parent is not None or "CASCADE" in cve_id)

                    level = "CRITICAL" if sev >= 9.0 else "HIGH" if sev >= 7.0 else "MEDIUM" if sev >= 4.0 else "LOW"

                    rec = VulnerabilityRecord(
                        vuln_id=cve_id,
                        package_name=pkg["name"],
                        ecosystem=ecosystem_stream.ecosystem,
                        severity_score=sev,
                        severity_level=level,
                        disclosure_lag_days=lag,
                        published_date=snapshot.timestamp,
                        is_transitive_cascade=is_casc,
                        parent_source_package=parent,
                    )
                    self.add_record(rec)
                    count += 1
        return count

    def to_polars(self) -> pl.DataFrame:
        """Convert collected vulnerabilities into a structured Polars DataFrame."""
        rows = [
            {
                "vuln_id": r.vuln_id,
                "package_name": r.package_name,
                "ecosystem": r.ecosystem,
                "severity_score": r.severity_score,
                "severity_level": r.severity_level,
                "disclosure_lag_days": r.disclosure_lag_days,
                "published_date": r.published_date,
                "is_transitive_cascade": int(r.is_transitive_cascade),
                "parent_source_package": r.parent_source_package or "",
            }
            for r in self.records
        ]
        if not rows:
            return pl.DataFrame(schema={
                "vuln_id": pl.Utf8,
                "package_name": pl.Utf8,
                "ecosystem": pl.Utf8,
                "severity_score": pl.Float64,
                "severity_level": pl.Utf8,
                "disclosure_lag_days": pl.Int32,
                "published_date": pl.Utf8,
                "is_transitive_cascade": pl.Int32,
                "parent_source_package": pl.Utf8,
            })
        return pl.DataFrame(rows)

    def get_mean_advisory_lag(self) -> float:
        """Compute the empirical mean disclosure lag across records."""
        if not self.records:
            return 0.0
        return sum(r.disclosure_lag_days for r in self.records) / len(self.records)
