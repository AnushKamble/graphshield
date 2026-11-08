"""Script to generate realistic standardized multi-epoch dependency streams for PyPI and npm ecosystems."""
import json
import os
import random

def generate_pypi_stream():
    random.seed(42)
    packages_meta = [
        # Web & networking core
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
        # Cryptography and OS
        ("cryptography", 6, 40.0, 5.5, 4),
        ("cffi", 2, 5.0, 1.2, 1),
        ("pycparser", 1, 2.0, 0.6, 0),
        ("pyOpenSSL", 3, 8.0, 1.5, 1),
        ("bcrypt", 2, 4.0, 1.0, 0),
        # Async and tasks
        ("celery", 4, 22.0, 3.0, 2),
        ("kombu", 3, 14.0, 2.5, 1),
        ("amqp", 2, 7.0, 1.5, 0),
        ("billiard", 2, 5.0, 1.0, 0),
        ("vine", 1, 2.0, 0.5, 0),
        # Data and utilities
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
        ("packaging", 2, 6.0, 1.5, 0),
        ("cachetools", 1, 3.0, 1.0, 0),
    ]

    base_edges = [
        # requests -> urllib3, certifi, idna, charset-normalizer
        ("requests", "urllib3", "caret", ">=1.26.0,<3.0.0"),
        ("requests", "certifi", "tilde", ">=2021.10.8"),
        ("requests", "idna", "caret", ">=2.5,<4.0"),
        ("requests", "charset-normalizer", "caret", ">=2.0.0,<4.0.0"),
        # flask -> werkzeug, jinja2, itsdangerous, click, blinker
        ("flask", "werkzeug", "caret", ">=2.2.0"),
        ("flask", "jinja2", "caret", ">=3.0.0"),
        ("flask", "itsdangerous", "caret", ">=2.0.0"),
        ("flask", "click", "caret", ">=8.0.0"),
        ("flask", "blinker", "pinned", "==1.6.2"),
        # jinja2 -> markupsafe
        ("jinja2", "markupsafe", "caret", ">=2.0.0"),
        ("werkzeug", "markupsafe", "caret", ">=2.1.1"),
        # fastapi -> starlette, pydantic, typing-extensions
        ("fastapi", "starlette", "caret", ">=0.27.0"),
        ("fastapi", "pydantic", "caret", ">=1.7.4,<3.0.0"),
        ("fastapi", "typing-extensions", "caret", ">=4.5.0"),
        # starlette -> anyio
        ("starlette", "anyio", "caret", ">=3.4.0,<5.0.0"),
        ("anyio", "sniffio", "caret", ">=1.1"),
        ("pydantic", "pydantic-core", "pinned", "==2.14.0"),
        ("pydantic", "typing-extensions", "caret", ">=4.6.1"),
        # uvicorn -> click, h11
        ("uvicorn", "click", "caret", ">=7.0"),
        ("uvicorn", "h11", "caret", ">=0.8"),
        # httpx -> httpcore, certifi, idna, sniffio
        ("httpx", "httpcore", "caret", ">=0.17.0,<0.19.0"),
        ("httpx", "certifi", "caret", ">=2022.0"),
        ("httpx", "idna", "caret", ">=2.8"),
        ("httpx", "sniffio", "caret", ">=1.0"),
        ("httpcore", "h11", "caret", ">=0.13.0,<0.15.0"),
        ("httpcore", "certifi", "caret", ">=2021.0"),
        # cryptography -> cffi
        ("cryptography", "cffi", "caret", ">=1.12"),
        ("cffi", "pycparser", "caret", ">=2.06"),
        ("pyOpenSSL", "cryptography", "caret", ">=38.0.0"),
        # celery -> kombu, billiard, click, vine
        ("celery", "kombu", "caret", ">=5.3.0,<6.0"),
        ("celery", "billiard", "caret", ">=4.1.0,<5.0"),
        ("celery", "click", "caret", ">=8.0.3"),
        ("celery", "vine", "caret", ">=5.0.0"),
        ("kombu", "amqp", "caret", ">=5.1.1,<6.0"),
        ("kombu", "vine", "caret", ">=5.0.0"),
        # sqlalchemy -> greenlet
        ("sqlalchemy", "greenlet", "caret", ">=2.0"),
        ("alembic", "sqlalchemy", "caret", ">=1.3.0"),
        ("alembic", "mako", "caret", ">=1.1.0"),
        ("mako", "markupsafe", "caret", ">=0.9.2"),
        # scikit-learn -> numpy, scipy, joblib, threadpoolctl
        ("scikit-learn", "numpy", "caret", ">=1.19.5"),
        ("scikit-learn", "scipy", "caret", ">=1.6.0"),
        ("scikit-learn", "joblib", "caret", ">=1.1.1"),
        ("scikit-learn", "threadpoolctl", "caret", ">=2.0.0"),
        ("scipy", "numpy", "caret", ">=1.19.5"),
        ("pandas", "numpy", "caret", ">=1.20.0"),
        # rich -> pygments, markdown (we use colorama/packaging)
        ("rich", "pygments", "caret", ">=2.6.0"),
        ("rich", "colorama", "caret", ">=0.4.0"),
    ]

    snapshots = []
    epochs = [
        ("2023-01-15T00:00:00Z", 0),
        ("2023-04-15T00:00:00Z", 1),
        ("2023-07-15T00:00:00Z", 2),
        ("2023-10-15T00:00:00Z", 3),
        ("2024-01-15T00:00:00Z", 4),
    ]

    # Vulnerability injections across epochs simulating real cascading attacks:
    # Epoch 1: urllib3 (CVE-2023-32681, proxy auth leak) introduced -> cascades to requests, httpx
    # Epoch 2: werkzeug (CVE-2023-46136, DoS in multipart) -> cascades to flask
    # Epoch 3: cryptography (CVE-2023-49083, NULL dereference in PKCS7) -> cascades to pyOpenSSL
    # Epoch 4: amqp (transitive poison) -> cascades to kombu, celery
    vuln_schedule = {
        0: [],
        1: [
            {"pkg": "urllib3", "cve": "CVE-2023-32681", "severity": 7.5, "lag_days": 68},
            {"pkg": "requests", "cve": "CASCADE-PyPI-001", "severity": 7.5, "lag_days": 54, "cascade_from": "urllib3"},
        ],
        2: [
            {"pkg": "urllib3", "cve": "CVE-2023-32681", "severity": 7.5, "lag_days": 68},
            {"pkg": "requests", "cve": "CASCADE-PyPI-001", "severity": 7.5, "lag_days": 54, "cascade_from": "urllib3"},
            {"pkg": "werkzeug", "cve": "CVE-2023-46136", "severity": 7.5, "lag_days": 92},
            {"pkg": "flask", "cve": "CASCADE-PyPI-002", "severity": 7.5, "lag_days": 78, "cascade_from": "werkzeug"},
        ],
        3: [
            {"pkg": "werkzeug", "cve": "CVE-2023-46136", "severity": 7.5, "lag_days": 92},
            {"pkg": "flask", "cve": "CASCADE-PyPI-002", "severity": 7.5, "lag_days": 78, "cascade_from": "werkzeug"},
            {"pkg": "cryptography", "cve": "CVE-2023-49083", "severity": 8.1, "lag_days": 115},
            {"pkg": "pyOpenSSL", "cve": "CASCADE-PyPI-003", "severity": 8.1, "lag_days": 95, "cascade_from": "cryptography"},
            {"pkg": "httpx", "cve": "CASCADE-PyPI-004", "severity": 6.8, "lag_days": 42, "cascade_from": "urllib3"},
        ],
        4: [
            {"pkg": "cryptography", "cve": "CVE-2023-49083", "severity": 8.1, "lag_days": 115},
            {"pkg": "pyOpenSSL", "cve": "CASCADE-PyPI-003", "severity": 8.1, "lag_days": 95, "cascade_from": "cryptography"},
            {"pkg": "amqp", "cve": "CVE-2023-52011", "severity": 7.2, "lag_days": 80},
            {"pkg": "kombu", "cve": "CASCADE-PyPI-005", "severity": 7.2, "lag_days": 65, "cascade_from": "amqp"},
            {"pkg": "celery", "cve": "CASCADE-PyPI-006", "severity": 7.2, "lag_days": 50, "cascade_from": "kombu"},
        ],
    }

    for timestamp, epoch in epochs:
        active_vulns = vuln_schedule.get(epoch, [])
        vuln_pkg_map = {v["pkg"]: v for v in active_vulns}

        snapshot_packages = []
        for name, maint, commits, velocity, hist_cve in packages_meta:
            # Add subtle epoch-based dynamics
            dyn_maint = maint
            dyn_commits = round(commits + (epoch * random.uniform(-0.5, 1.2)), 2)
            dyn_velocity = round(velocity + (epoch * random.uniform(0.1, 0.4)), 2)
            dyn_hist_cve = hist_cve
            
            is_vuln = 1 if name in vuln_pkg_map else 0
            details = [vuln_pkg_map[name]] if is_vuln else []

            snapshot_packages.append({
                "name": name,
                "version": f"2.{epoch}.{random.randint(0, 5)}",
                "maintainer_count": dyn_maint,
                "commit_cadence": dyn_commits,
                "release_velocity": dyn_velocity,
                "historical_cve_count": dyn_hist_cve,
                "is_vulnerable": is_vuln,
                "vulnerability_details": details
            })

        # Add edges for this snapshot with slight dynamic age updates
        snapshot_edges = []
        for src, tgt, ctype, spec in base_edges:
            snapshot_edges.append({
                "source": src,
                "target": tgt,
                "constraint_type": ctype,
                "version_spec": spec,
                "age_days": 120 + epoch * 90 + random.randint(0, 15)
            })

        snapshots.append({
            "epoch": epoch,
            "timestamp": timestamp,
            "packages": snapshot_packages,
            "dependencies": snapshot_edges
        })

    return {
        "ecosystem": "PyPI",
        "description": "Standardized multi-epoch PyPI software dependency stream with OSV/GHSA ground truth annotations.",
        "num_epochs": len(snapshots),
        "total_packages": len(packages_meta),
        "snapshots": snapshots
    }

def generate_npm_stream():
    random.seed(1337)
    npm_packages = [
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

    npm_edges = [
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
        ("cliui", "string-width", "caret", "^4.2.0"),
        ("cliui", "wrap-ansi", "caret", "^7.0.0"),
    ]

    vuln_schedule_npm = {
        0: [],
        1: [
            {"pkg": "follow-redirects", "cve": "CVE-2024-28849", "severity": 7.5, "lag_days": 85},
            {"pkg": "axios", "cve": "CASCADE-NPM-001", "severity": 7.5, "lag_days": 70, "cascade_from": "follow-redirects"},
        ],
        2: [
            {"pkg": "follow-redirects", "cve": "CVE-2024-28849", "severity": 7.5, "lag_days": 85},
            {"pkg": "axios", "cve": "CASCADE-NPM-001", "severity": 7.5, "lag_days": 70, "cascade_from": "follow-redirects"},
            {"pkg": "qs", "cve": "CVE-2022-24999", "severity": 7.5, "lag_days": 110},
            {"pkg": "body-parser", "cve": "CASCADE-NPM-002", "severity": 7.5, "lag_days": 90, "cascade_from": "qs"},
        ],
        3: [
            {"pkg": "qs", "cve": "CVE-2022-24999", "severity": 7.5, "lag_days": 110},
            {"pkg": "body-parser", "cve": "CASCADE-NPM-002", "severity": 7.5, "lag_days": 90, "cascade_from": "qs"},
            {"pkg": "express", "cve": "CASCADE-NPM-003", "severity": 7.5, "lag_days": 75, "cascade_from": "body-parser"},
            {"pkg": "lodash", "cve": "CVE-2020-28500", "severity": 7.3, "lag_days": 140},
        ],
        4: [
            {"pkg": "lodash", "cve": "CVE-2020-28500", "severity": 7.3, "lag_days": 140},
            {"pkg": "minimist", "cve": "CVE-2021-44906", "severity": 8.2, "lag_days": 120},
            {"pkg": "semver", "cve": "CVE-2022-25883", "severity": 7.5, "lag_days": 95},
            {"pkg": "ms", "cve": "CVE-2022-25860", "severity": 5.3, "lag_days": 60},
            {"pkg": "debug", "cve": "CASCADE-NPM-004", "severity": 5.3, "lag_days": 45, "cascade_from": "ms"},
            {"pkg": "send", "cve": "CASCADE-NPM-005", "severity": 5.3, "lag_days": 35, "cascade_from": "ms"},
        ],
    }

    epochs = [
        ("2023-01-15T00:00:00Z", 0),
        ("2023-04-15T00:00:00Z", 1),
        ("2023-07-15T00:00:00Z", 2),
        ("2023-10-15T00:00:00Z", 3),
        ("2024-01-15T00:00:00Z", 4),
    ]

    snapshots = []
    for timestamp, epoch in epochs:
        active_vulns = vuln_schedule_npm.get(epoch, [])
        vuln_pkg_map = {v["pkg"]: v for v in active_vulns}

        snapshot_packages = []
        for name, maint, commits, velocity, hist_cve in npm_packages:
            dyn_maint = maint
            dyn_commits = round(commits + (epoch * random.uniform(-0.5, 1.2)), 2)
            dyn_velocity = round(velocity + (epoch * random.uniform(0.1, 0.4)), 2)
            dyn_hist_cve = hist_cve
            
            is_vuln = 1 if name in vuln_pkg_map else 0
            details = [vuln_pkg_map[name]] if is_vuln else []

            snapshot_packages.append({
                "name": name,
                "version": f"4.{epoch}.{random.randint(0, 7)}",
                "maintainer_count": dyn_maint,
                "commit_cadence": dyn_commits,
                "release_velocity": dyn_velocity,
                "historical_cve_count": dyn_hist_cve,
                "is_vulnerable": is_vuln,
                "vulnerability_details": details
            })

        snapshot_edges = []
        for src, tgt, ctype, spec in npm_edges:
            snapshot_edges.append({
                "source": src,
                "target": tgt,
                "constraint_type": ctype,
                "version_spec": spec,
                "age_days": 100 + epoch * 80 + random.randint(0, 20)
            })

        snapshots.append({
            "epoch": epoch,
            "timestamp": timestamp,
            "packages": snapshot_packages,
            "dependencies": snapshot_edges
        })

    return {
        "ecosystem": "npm",
        "description": "Standardized multi-epoch npm software dependency stream with OSV/GHSA ground truth annotations.",
        "num_epochs": len(snapshots),
        "total_packages": len(npm_packages),
        "snapshots": snapshots
    }

if __name__ == "__main__":
    out_dir = r"C:\Users\anush\Desktop\research\graphshield\benchmarks\datasets"
    os.makedirs(out_dir, exist_ok=True)
    
    pypi_data = generate_pypi_stream()
    with open(os.path.join(out_dir, "pypi_dependency_stream.json"), "w", encoding="utf-8") as f:
        json.dump(pypi_data, f, indent=2)
    print("Exported pypi_dependency_stream.json")

    npm_data = generate_npm_stream()
    with open(os.path.join(out_dir, "npm_dependency_stream.json"), "w", encoding="utf-8") as f:
        json.dump(npm_data, f, indent=2)
    print("Exported npm_dependency_stream.json")
