"""
Script to generate realistic standardized multi-epoch dependency streams
for PyPI (1,200 packages) and npm (1,500 packages) ecosystems with
scale-free DAG topologies and cascading vulnerability trajectories.
"""

import json
import os
import random
from typing import Dict, List, Tuple
import numpy as np


def build_scale_free_stream(
    ecosystem: str,
    total_packages: int,
    core_packages: List[Tuple[str, int, float, float, int]],
    core_edges: List[Tuple[str, str, str, str]],
    seed_vulns: List[Tuple[str, str, float, int, int]],  # (pkg, cve, cvss, start_epoch, lag_days)
    random_seed: int = 42,
) -> Dict:
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 1. Establish full package universe
    packages_meta: List[Tuple[str, int, float, float, int]] = list(core_packages)
    existing_count = len(packages_meta)

    prefix = ecosystem.lower()
    for i in range(existing_count, total_packages):
        name = f"{prefix}-pkg-{i:04d}"
        maint = max(1, int(np.random.exponential(2.0)))
        commits = round(float(np.random.gamma(3.0, 3.5)), 2)
        velocity = round(float(np.random.gamma(2.0, 1.2)), 2)
        hist_cve = 1 if (random.random() < 0.12) else 0
        packages_meta.append((name, maint, commits, velocity, hist_cve))

    pkg_names = [p[0] for p in packages_meta]
    pkg_index_map = {name: idx for idx, name in enumerate(pkg_names)}

    # 2. Build Scale-Free Directed Acyclic Graph (DAG) starting with core edges
    degrees = [1] * total_packages
    edges: List[Tuple[str, str, str, str]] = list(core_edges)

    for src, tgt, _, _ in edges:
        if tgt in pkg_index_map:
            degrees[pkg_index_map[tgt]] += 10

    # Boost core packages to be heavy foundational hubs
    for idx in range(min(existing_count, 30)):
        degrees[idx] += 30

    constraint_types = (
        ["caret"] * 50 + ["tilde"] * 25 + ["pinned"] * 18 + ["wildcard"] * 7
        if ecosystem == "npm"
        else ["caret"] * 45 + ["tilde"] * 25 + ["pinned"] * 22 + ["wildcard"] * 8
    )

    # Add preferential attachment dependencies for packages beyond existing_count
    for i in range(existing_count, total_packages):
        m = random.randint(1, min(4, i))
        probs = np.array(degrees[:i], dtype=np.float64)
        probs /= probs.sum()
        targets = np.random.choice(i, size=m, replace=False, p=probs)
        for t in targets:
            ctype = random.choice(constraint_types)
            spec = f">={t}.0.0" if ctype != "pinned" else f"=={t}.0.0"
            edges.append((pkg_names[i], pkg_names[t], ctype, spec))
            degrees[t] += 1

    # 3. Simulate Multi-Epoch Temporal Dynamics and Cascades
    epochs = [
        ("2023-01-15T00:00:00Z", 0),
        ("2023-04-15T00:00:00Z", 1),
        ("2023-07-15T00:00:00Z", 2),
        ("2023-10-15T00:00:00Z", 3),
        ("2024-01-15T00:00:00Z", 4),
    ]

    # Map upstream target -> list of (downstream source, constraint)
    adjacency: Dict[str, List[Tuple[str, str]]] = {name: [] for name in pkg_names}
    for src, tgt, ctype, _ in edges:
        if tgt in adjacency:
            adjacency[tgt].append((src, ctype))

    snapshots = []
    for timestamp, epoch in epochs:
        active_vulns: Dict[str, Dict] = {}

        # A. Seed primary vulnerabilities
        for seed_pkg, cve, cvss, start_ep, lag in seed_vulns:
            if epoch >= start_ep:
                active_vulns[seed_pkg] = {
                    "pkg": seed_pkg,
                    "cve": cve,
                    "severity": cvss,
                    "lag_days": lag,
                }

        # B. Propagate cascading exposures along non-pinned edges
        visited_in_epoch = set(active_vulns.keys())
        current_wave = list(active_vulns.keys())

        for hop in range(2):
            next_wave = []
            for infected in current_wave:
                inf_meta = active_vulns[infected]
                for downstream, ctype in adjacency.get(infected, []):
                    if downstream not in visited_in_epoch:
                        if ctype != "pinned":
                            active_vulns[downstream] = {
                                "pkg": downstream,
                                "cve": f"CASCADE-{ecosystem}-{inf_meta['cve']}",
                                "severity": round(inf_meta["severity"] * 0.95, 1),
                                "lag_days": max(15, inf_meta["lag_days"] - 14 * (hop + 1)),
                                "cascade_from": infected,
                            }
                            visited_in_epoch.add(downstream)
                            next_wave.append(downstream)
            current_wave = next_wave

        # Build snapshot package list
        snapshot_packages = []
        for name, maint, commits, velocity, hist_cve in packages_meta:
            dyn_maint = maint
            dyn_commits = round(commits + (epoch * random.uniform(-0.3, 0.8)), 2)
            dyn_velocity = round(velocity + (epoch * random.uniform(0.05, 0.25)), 2)
            dyn_hist_cve = hist_cve

            is_vuln = 1 if name in active_vulns else 0
            details = [active_vulns[name]] if is_vuln else []

            snapshot_packages.append({
                "name": name,
                "version": f"2.{epoch}.{random.randint(0, 5)}",
                "maintainer_count": dyn_maint,
                "commit_cadence": max(0.5, dyn_commits),
                "release_velocity": max(0.2, dyn_velocity),
                "historical_cve_count": dyn_hist_cve,
                "is_vulnerable": is_vuln,
                "vulnerability_details": details,
            })

        # Build snapshot edges with dynamic ages
        snapshot_edges = []
        for src, tgt, ctype, spec in edges:
            snapshot_edges.append({
                "source": src,
                "target": tgt,
                "constraint_type": ctype,
                "version_spec": spec,
                "age_days": 90 + epoch * 90 + random.randint(0, 15),
            })

        snapshots.append({
            "epoch": epoch,
            "timestamp": timestamp,
            "packages": snapshot_packages,
            "dependencies": snapshot_edges,
        })

    return {
        "ecosystem": ecosystem,
        "description": f"Standardized scale-free multi-epoch {ecosystem} dependency stream ({total_packages} packages, {len(edges)} dependencies) with ground-truth vulnerability cascades.",
        "num_epochs": len(snapshots),
        "total_packages": total_packages,
        "total_dependencies": len(edges),
        "snapshots": snapshots,
    }


def generate_pypi_stream() -> Dict:
    core_pypi = [
        ("requests", 3, 14.5, 3.2, 2),
        ("urllib3", 4, 22.0, 4.0, 3),
        ("certifi", 2, 5.0, 2.0, 1),
        ("idna", 2, 4.0, 1.5, 0),
        ("charset-normalizer", 1, 8.0, 2.5, 1),
        ("flask", 5, 18.0, 3.0, 2),
        ("werkzeug", 4, 16.0, 3.5, 3),
        ("jinja2", 3, 11.0, 2.0, 1),
        ("itsdangerous", 2, 3.0, 1.0, 0),
        ("click", 4, 12.0, 2.2, 0),
        ("blinker", 1, 2.0, 0.8, 0),
        ("markupsafe", 2, 6.0, 1.8, 0),
        ("fastapi", 4, 35.0, 6.5, 1),
        ("starlette", 3, 20.0, 4.5, 1),
        ("pydantic", 5, 45.0, 8.0, 2),
        ("pydantic-core", 3, 25.0, 5.0, 0),
        ("typing-extensions", 2, 10.0, 3.0, 0),
        ("anyio", 2, 15.0, 3.8, 0),
        ("sniffio", 1, 2.0, 0.5, 0),
        ("uvicorn", 3, 16.0, 3.2, 1),
        ("httpx", 3, 24.0, 4.2, 1),
        ("httpcore", 2, 14.0, 3.0, 0),
        ("h11", 2, 3.0, 1.0, 0),
        ("cryptography", 6, 40.0, 5.5, 4),
        ("cffi", 2, 5.0, 1.2, 1),
        ("pycparser", 1, 2.0, 0.6, 0),
        ("pyOpenSSL", 3, 8.0, 1.5, 1),
        ("bcrypt", 2, 4.0, 1.0, 0),
        ("celery", 4, 22.0, 3.0, 2),
        ("kombu", 3, 14.0, 2.5, 1),
        ("amqp", 2, 7.0, 1.5, 0),
        ("billiard", 2, 5.0, 1.0, 0),
        ("vine", 1, 2.0, 0.5, 0),
        ("sqlalchemy", 5, 32.0, 4.8, 2),
        ("greenlet", 2, 6.0, 1.5, 0),
        ("alembic", 3, 12.0, 2.4, 0),
        ("mako", 2, 4.0, 1.0, 0),
        ("pandas", 8, 65.0, 6.0, 3),
        ("numpy", 7, 70.0, 5.0, 2),
        ("scipy", 6, 50.0, 4.0, 1),
        ("scikit-learn", 7, 48.0, 3.8, 1),
        ("joblib", 2, 6.0, 1.2, 0),
        ("threadpoolctl", 2, 3.0, 0.8, 0),
        ("tqdm", 2, 7.0, 2.0, 0),
        ("rich", 3, 28.0, 5.0, 0),
        ("pygments", 3, 8.0, 1.5, 1),
        ("colorama", 1, 2.0, 0.5, 0),
        ("psutil", 3, 9.0, 1.8, 0),
        ("pytest", 4, 25.0, 4.0, 1),
        ("attrs", 2, 10.0, 2.0, 0),
    ]

    pypi_core_edges = [
        ("requests", "urllib3", "caret", ">=1.26.0,<3.0.0"),
        ("requests", "certifi", "tilde", ">=2021.10.8"),
        ("requests", "idna", "caret", ">=2.5,<4.0"),
        ("requests", "charset-normalizer", "caret", ">=2.0.0,<4.0.0"),
        ("flask", "werkzeug", "caret", ">=2.2.0"),
        ("flask", "jinja2", "caret", ">=3.0.0"),
        ("flask", "itsdangerous", "caret", ">=2.0.0"),
        ("flask", "click", "caret", ">=8.0.0"),
        ("flask", "blinker", "pinned", "==1.6.2"),
        ("jinja2", "markupsafe", "caret", ">=2.0.0"),
        ("werkzeug", "markupsafe", "caret", ">=2.1.1"),
        ("fastapi", "starlette", "caret", ">=0.27.0"),
        ("fastapi", "pydantic", "caret", ">=1.7.4,<3.0.0"),
        ("fastapi", "typing-extensions", "caret", ">=4.5.0"),
        ("starlette", "anyio", "caret", ">=3.4.0,<5.0.0"),
        ("anyio", "sniffio", "caret", ">=1.1"),
        ("pydantic", "pydantic-core", "pinned", "==2.14.0"),
        ("pydantic", "typing-extensions", "caret", ">=4.6.1"),
        ("uvicorn", "click", "caret", ">=7.0"),
        ("uvicorn", "h11", "caret", ">=0.8"),
        ("httpx", "httpcore", "caret", ">=0.17.0,<0.19.0"),
        ("httpx", "certifi", "caret", ">=2022.0"),
        ("httpx", "idna", "caret", ">=2.8"),
        ("httpx", "sniffio", "caret", ">=1.0"),
        ("httpcore", "h11", "caret", ">=0.13.0,<0.15.0"),
        ("httpcore", "certifi", "caret", ">=2021.0"),
        ("cryptography", "cffi", "caret", ">=1.12"),
        ("cffi", "pycparser", "caret", ">=2.06"),
        ("pyOpenSSL", "cryptography", "caret", ">=38.0.0"),
        ("celery", "kombu", "caret", ">=5.3.0,<6.0"),
        ("celery", "billiard", "caret", ">=4.1.0,<5.0"),
        ("celery", "click", "caret", ">=8.0.3"),
        ("celery", "vine", "caret", ">=5.0.0"),
        ("kombu", "amqp", "caret", ">=5.1.1,<6.0"),
        ("kombu", "vine", "caret", ">=5.0.0"),
        ("sqlalchemy", "greenlet", "caret", ">=2.0"),
        ("alembic", "sqlalchemy", "caret", ">=1.3.0"),
        ("alembic", "mako", "caret", ">=1.1.0"),
        ("mako", "markupsafe", "caret", ">=0.9.2"),
        ("scikit-learn", "numpy", "caret", ">=1.19.5"),
        ("scikit-learn", "scipy", "caret", ">=1.6.0"),
        ("scikit-learn", "joblib", "caret", ">=1.1.1"),
        ("scikit-learn", "threadpoolctl", "caret", ">=2.0.0"),
        ("scipy", "numpy", "caret", ">=1.19.5"),
        ("pandas", "numpy", "caret", ">=1.20.0"),
        ("rich", "pygments", "caret", ">=2.6.0"),
        ("rich", "colorama", "caret", ">=0.4.0"),
    ]

    pypi_seed_vulns = [
        ("urllib3", "CVE-2023-32681", 7.5, 1, 68),
        ("werkzeug", "CVE-2023-46136", 7.5, 2, 92),
        ("cryptography", "CVE-2023-49083", 8.1, 3, 115),
        ("amqp", "CVE-2023-52011", 7.2, 3, 80),
        ("jinja2", "CVE-2024-22195", 6.5, 4, 45),
        ("pydantic", "CVE-2024-37890", 7.8, 2, 60),
    ]

    return build_scale_free_stream(
        ecosystem="PyPI",
        total_packages=1200,
        core_packages=core_pypi,
        core_edges=pypi_core_edges,
        seed_vulns=pypi_seed_vulns,
        random_seed=42,
    )


def generate_npm_stream() -> Dict:
    core_npm = [
        ("express", 5, 25.0, 4.0, 3),
        ("body-parser", 3, 10.0, 2.0, 2),
        ("cookie-parser", 2, 6.0, 1.5, 1),
        ("serve-static", 2, 8.0, 1.8, 1),
        ("send", 3, 12.0, 2.2, 2),
        ("debug", 3, 15.0, 3.0, 1),
        ("ms", 1, 4.0, 1.0, 1),
        ("mime", 2, 5.0, 1.2, 1),
        ("fresh", 1, 3.0, 0.8, 1),
        ("range-parser", 2, 4.0, 1.0, 0),
        ("etag", 2, 3.0, 0.8, 0),
        ("destroy", 1, 2.0, 0.5, 0),
        ("encodeurl", 1, 2.0, 0.5, 0),
        ("escape-html", 1, 2.0, 0.5, 0),
        ("qs", 3, 16.0, 3.5, 3),
        ("content-type", 1, 3.0, 0.8, 0),
        ("raw-body", 2, 6.0, 1.5, 1),
        ("bytes", 1, 3.0, 0.8, 0),
        ("http-errors", 2, 8.0, 2.0, 0),
        ("inherits", 1, 1.0, 0.2, 0),
        ("statuses", 2, 4.0, 1.0, 0),
        ("unpipe", 1, 2.0, 0.5, 0),
        ("lodash", 4, 30.0, 4.5, 5),
        ("axios", 5, 45.0, 6.0, 4),
        ("follow-redirects", 2, 12.0, 2.5, 3),
        ("form-data", 3, 14.0, 2.8, 1),
        ("combined-stream", 1, 3.0, 0.8, 0),
        ("delayed-stream", 1, 2.0, 0.5, 0),
        ("chalk", 4, 20.0, 3.5, 0),
        ("ansi-styles", 2, 6.0, 1.5, 0),
        ("supports-color", 2, 7.0, 1.8, 0),
        ("semver", 4, 18.0, 3.0, 2),
        ("minimist", 2, 5.0, 1.0, 2),
        ("async", 3, 10.0, 2.0, 1),
        ("commander", 4, 22.0, 4.0, 0),
        ("glob", 3, 12.0, 2.5, 1),
        ("minimatch", 2, 8.0, 2.0, 1),
        ("brace-expansion", 2, 4.0, 1.0, 0),
        ("balanced-match", 1, 2.0, 0.5, 0),
        ("concat-map", 1, 1.0, 0.2, 0),
        ("fs-extra", 3, 14.0, 2.5, 0),
        ("graceful-fs", 2, 5.0, 1.0, 0),
        ("jsonfile", 2, 4.0, 1.0, 0),
        ("universalify", 1, 2.0, 0.5, 0),
        ("uuid", 3, 16.0, 3.0, 0),
        ("ws", 4, 28.0, 4.5, 3),
        ("yargs", 4, 24.0, 3.8, 1),
        ("cliui", 2, 5.0, 1.2, 0),
        ("wrap-ansi", 2, 7.0, 1.5, 0),
        ("string-width", 2, 8.0, 1.8, 0),
    ]

    npm_core_edges = [
        ("express", "body-parser", "caret", "^1.20.1"),
        ("express", "cookie-parser", "caret", "~1.4.6"),
        ("express", "serve-static", "caret", "1.15.0"),
        ("express", "send", "caret", "^0.18.0"),
        ("express", "debug", "caret", "^2.6.9"),
        ("express", "qs", "caret", "^6.11.0"),
        ("express", "etag", "caret", "~1.8.1"),
        ("express", "fresh", "caret", "0.5.2"),
        ("express", "encodeurl", "caret", "~1.0.2"),
        ("express", "escape-html", "caret", "~1.0.3"),
        ("body-parser", "bytes", "caret", "3.1.2"),
        ("body-parser", "content-type", "caret", "~1.0.4"),
        ("body-parser", "debug", "caret", "^2.6.9"),
        ("body-parser", "http-errors", "caret", "^2.0.0"),
        ("body-parser", "raw-body", "caret", "^2.5.1"),
        ("body-parser", "qs", "caret", "^6.11.0"),
        ("send", "debug", "caret", "^2.6.9"),
        ("send", "destroy", "caret", "^1.2.0"),
        ("send", "etag", "caret", "~1.8.1"),
        ("send", "fresh", "caret", "0.5.2"),
        ("send", "http-errors", "caret", "^2.0.0"),
        ("send", "mime", "caret", "^1.6.0"),
        ("send", "ms", "caret", "2.1.3"),
        ("send", "range-parser", "caret", "~1.2.1"),
        ("send", "statuses", "caret", "^2.0.1"),
        ("serve-static", "send", "caret", "^0.18.0"),
        ("raw-body", "bytes", "caret", "3.1.2"),
        ("raw-body", "http-errors", "caret", "^2.0.0"),
        ("raw-body", "unpipe", "caret", "~1.0.0"),
        ("debug", "ms", "caret", "^2.1.2"),
        ("axios", "follow-redirects", "caret", "^1.15.0"),
        ("axios", "form-data", "caret", "^4.0.0"),
        ("form-data", "combined-stream", "caret", "^1.0.8"),
        ("form-data", "delayed-stream", "caret", "~1.0.0"),
        ("form-data", "mime", "caret", "^2.5.0"),
        ("chalk", "ansi-styles", "caret", "^4.1.0"),
        ("chalk", "supports-color", "caret", "^7.1.0"),
        ("glob", "minimatch", "caret", "^3.0.4"),
        ("minimatch", "brace-expansion", "caret", "^1.1.7"),
        ("brace-expansion", "balanced-match", "caret", "^1.0.0"),
        ("brace-expansion", "concat-map", "caret", "0.0.1"),
        ("fs-extra", "graceful-fs", "caret", "^4.2.0"),
        ("fs-extra", "jsonfile", "caret", "^6.0.1"),
        ("fs-extra", "universalify", "caret", "^2.0.0"),
        ("yargs", "cliui", "caret", "^7.0.2"),
    ]

    npm_seed_vulns = [
        ("lodash", "CVE-2023-26159", 7.8, 1, 85),
        ("axios", "CVE-2023-45857", 7.5, 2, 64),
        ("follow-redirects", "CVE-2024-28849", 7.0, 3, 52),
        ("ws", "CVE-2024-37890", 7.5, 2, 70),
        ("semver", "CVE-2022-25883", 6.8, 3, 90),
        ("minimist", "CVE-2021-44906", 8.2, 4, 110),
    ]

    return build_scale_free_stream(
        ecosystem="npm",
        total_packages=1500,
        core_packages=core_npm,
        core_edges=npm_core_edges,
        seed_vulns=npm_seed_vulns,
        random_seed=1337,
    )


if __name__ == "__main__":
    out_dir = r"C:\Users\anush\Desktop\research\graphshield\benchmarks\datasets"
    os.makedirs(out_dir, exist_ok=True)

    print("[*] Generating PyPI scale-free dependency stream (1,200 packages)...")
    pypi_data = generate_pypi_stream()
    with open(os.path.join(out_dir, "pypi_dependency_stream.json"), "w", encoding="utf-8") as f:
        json.dump(pypi_data, f, indent=2)
    print(f"[+] Exported pypi_dependency_stream.json ({pypi_data['total_packages']} pkgs, {pypi_data['total_dependencies']} edges)")

    print("[*] Generating npm scale-free dependency stream (1,500 packages)...")
    npm_data = generate_npm_stream()
    with open(os.path.join(out_dir, "npm_dependency_stream.json"), "w", encoding="utf-8") as f:
        json.dump(npm_data, f, indent=2)
    print(f"[+] Exported npm_dependency_stream.json ({npm_data['total_packages']} pkgs, {npm_data['total_dependencies']} edges)")
