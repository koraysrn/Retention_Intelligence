"""End-to-end pipeline orchestration (Phase 1+).

Phases:
  0. Data quality check
  1. dbt transformation
  2. Model training + evaluation
  3. Batch scoring + A/B assignment
  4. Agent workflow (high-risk segment)
"""

from __future__ import annotations

import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STEPS = [
    ("faz0", [sys.executable, "-m", "scripts.faz0_data_quality"]),
    ("dbt-run", ["dbt", "run", "--project-dir", "dbt", "--profiles-dir", "dbt"]),
    ("train", [sys.executable, "-m", "src.models.train"]),
    ("score", [sys.executable, "-m", "src.serving.batch_score"]),
    ("agents", [sys.executable, "-m", "src.agents.orchestrator"]),
]


def run_step(name: str, cmd: list[str]) -> bool:
    logger.info("Step started: %s -> %s", name, " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("Step failed: %s (exit=%d)", name, result.returncode)
        return False
    return True


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, cmd in STEPS:
        if only and name != only:
            continue
        if not run_step(name, cmd):
            sys.exit(1)
    logger.info("Pipeline completed.")


if __name__ == "__main__":
    main()
