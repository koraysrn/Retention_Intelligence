"""Dependency version audit — queries the latest stable versions from PyPI,
GitHub Actions and Docker Hub.

Purpose: list up-to-date stable versions of every component in a single command
to support the "zero version mismatch" policy.

Usage: python -m scripts.audit_versions
"""

from __future__ import annotations

import json
import urllib.request

USER_AGENT = "version-audit/1.0"


def fetch_json(url: str, headers: dict | None = None) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


PYPI_PACKAGES = [
    # core
    "pandas",
    "numpy",
    "pyarrow",
    "duckdb",
    "pyyaml",
    "pydantic",
    "pydantic-settings",
    "python-dotenv",
    # ml
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "catboost",
    "imbalanced-learn",
    "shap",
    "matplotlib",
    "seaborn",
    # experiment
    "scipy",
    "statsmodels",
    # serving
    "fastapi",
    "uvicorn",
    # mlops
    "mlflow",
    # monitoring
    "evidently",
    # agents
    "langgraph",
    "langchain",
    "langchain-openai",
    "langchain-anthropic",
    "langchain-community",
    "psycopg",
    "pgvector",
    # dev
    "pytest",
    "pytest-cov",
    "ruff",
    "mypy",
    "pre-commit",
    # build
    "setuptools",
    # dbt
    "dbt-core",
    "dbt-duckdb",
]

GITHUB_REPOS = [
    # GitHub Actions
    "actions/checkout",
    "actions/setup-python",
    "astral-sh/setup-uv",
    "codecov/codecov-action",
    # pre-commit hook sources
    "astral-sh/ruff-pre-commit",
    "pre-commit/pre-commit-hooks",
    "pre-commit/mirrors-mypy",
    # source repo for the MLflow docker image
    "mlflow/mlflow",
]

DOCKER_IMAGES = [
    "pgvector/pgvector",
    "minio/minio",
    "minio/mc",
]


def audit_pypi() -> None:
    print("=" * 70)
    print("PYPI — latest stable versions")
    print("=" * 70)
    for pkg in PYPI_PACKAGES:
        try:
            data = fetch_json(f"https://pypi.org/pypi/{pkg}/json")
            info = data["info"]
            requires_python = info.get("requires_python") or "(none)"
            print(f"{pkg:24s} {info['version']:16s} requires_python={requires_python}")
        except Exception as exc:  # noqa: BLE001
            print(f"{pkg:24s} ERROR: {exc}")


def audit_github() -> None:
    print()
    print("=" * 70)
    print("GITHUB — actions / pre-commit / mlflow versions")
    print("=" * 70)
    for repo in GITHUB_REPOS:
        try:
            data = fetch_json(
                f"https://api.github.com/repos/{repo}/releases/latest",
                headers={"Accept": "application/vnd.github+json"},
            )
            print(f"{repo:32s} {data.get('tag_name', '(none)')}")
        except Exception as exc:  # noqa: BLE001
            print(f"{repo:32s} ERROR: {exc}")


def audit_docker() -> None:
    print()
    print("=" * 70)
    print("DOCKER HUB — image tags")
    print("=" * 70)
    for image in DOCKER_IMAGES:
        try:
            url = (
                f"https://hub.docker.com/v2/repositories/{image}/tags/"
                "?page_size=8&ordering=last_updated"
            )
            data = fetch_json(url)
            tags = [t["name"] for t in data.get("results", [])]
            print(f"{image:24s} {tags}")
        except Exception as exc:  # noqa: BLE001
            print(f"{image:24s} ERROR: {exc}")


def main() -> None:
    audit_pypi()
    audit_github()
    audit_docker()


if __name__ == "__main__":
    main()
