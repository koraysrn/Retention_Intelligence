# =============================================================================
# Churn Re-Engagement Platform — Frequently used commands
# On Windows cmd / PowerShell you can also use the `py -m ...` targets directly
# instead of `make`.
# =============================================================================
SHELL := /bin/bash
PYTHON ?= python

.PHONY: help install dev-lock lint format typecheck test train score ab-analyze ab-experiment run-api \
        dbt-run dbt-test infra-up infra-down agent-run faz0 ingest features

help: ## List commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install core dependencies
	$(PYTHON) -m pip install -e ".[ml,experiment,mlops,monitoring,dbt]"

dev-lock: ## Install development dependencies
	$(PYTHON) -m pip install -e ".[dev,agents,serving,dbt]"

lint: ## Ruff lint
	$(PYTHON) -m ruff check src scripts tests

format: ## Ruff format
	$(PYTHON) -m ruff format src scripts tests

typecheck: ## Mypy type check
	$(PYTHON) -m mypy src

test: ## Run tests
	$(PYTHON) -m pytest

faz0: ## Phase 0: data quality exploration
	$(PYTHON) -m scripts.faz0_data_quality

ingest: ## Phase 1: CSV -> DuckDB raw ingestion
	$(PYTHON) -m scripts.ingest

features: ## Phase 1: build the feature set from mart tables
	$(PYTHON) -m scripts.build_features

train: ## Churn ensemble training (XGBoost + LightGBM + CatBoost)
	$(PYTHON) -m scripts.train_ensemble

score: ## Batch risk scoring
	$(PYTHON) -m src.serving.batch_score

ab-analyze: ## A/B analysis (synthetic demo)
	$(PYTHON) -m src.experiments.analysis

ab-experiment: ## Phase 3: A/B experiment pipeline (sample + assignment + analysis)
	$(PYTHON) -m scripts.run_ab_experiment

run-api: ## Serving API
	$(PYTHON) -m uvicorn src.serving.api:app --reload

agent-run: ## Run the agent workflow
	$(PYTHON) -m src.agents.orchestrator

dbt-run: ## Run dbt models
	dbt run --project-dir dbt --profiles-dir dbt

dbt-test: ## dbt tests
	dbt test --project-dir dbt --profiles-dir dbt

infra-up: ## Start the local infrastructure
	docker compose up -d

infra-down: ## Stop the local infrastructure
	docker compose down
