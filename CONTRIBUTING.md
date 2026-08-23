# Contributing

Thank you for considering contributing to the Churn Re-Engagement Platform.
This document describes the workflow for reporting issues and submitting
changes.

## Table of contents

- [Contributing](#contributing)
  - [Table of contents](#table-of-contents)
  - [Code of conduct](#code-of-conduct)
  - [Getting started](#getting-started)
  - [Development workflow](#development-workflow)
  - [Code style](#code-style)
  - [Testing](#testing)
  - [Commit conventions](#commit-conventions)
  - [Pull requests](#pull-requests)

## Code of conduct

All contributors are expected to follow our
[Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

## Getting started

```bash
# 1. Clone and create a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS

# 2. Install the full development stack
python -m pip install -e ".[dev,ml,experiment,serving,mlops,monitoring,agents,dbt]"

# 3. Install pre-commit hooks
pre-commit install

# 4. Configure local environment
copy .env.example .env            # Windows cmd
# cp .env.example .env            # Linux/macOS

# 5. Start local infrastructure (optional)
docker compose up -d
```

## Development workflow

1. Open an issue first to discuss the change (bug report or feature request).
2. Create a feature branch from `main`.
3. Make focused, incremental commits.
4. Run the quality gates locally before pushing.
5. Open a pull request and fill in the PR template.

## Code style

The project uses [`ruff`](https://docs.astral.sh/ruff/) for linting and
formatting and [`mypy`](https://mypy-lang.org/) for type checking. Configuration
lives in [`pyproject.toml`](pyproject.toml).

```bash
make lint          # ruff check
make format        # ruff format
make typecheck     # mypy
```

The `pre-commit` configuration runs the same checks automatically on commit.

## Testing

```bash
make test          # full pytest suite with coverage
```

Tests are organized under [`tests/`](tests/) and mirror the `src/` layout. Add
or update tests whenever you change behavior.

## Commit conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):

```text
feat: add CUPED-based variance reduction
fix: handle empty batch in scoring API
docs: update architecture diagram
refactor: extract RFM feature builder
test: cover drift detection edge cases
ci: add frontend build job
```

## Pull requests

- Keep changes scoped to a single concern.
- Link the related issue with `Closes #<issue>`.
- Ensure CI (lint, typecheck, tests, frontend build) passes.
- Request a review and respond to feedback.
