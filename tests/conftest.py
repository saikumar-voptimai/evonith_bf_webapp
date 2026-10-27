"""Test bootstrap helpers for local package resolution, the test environment and service markers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("COKE_ENERGY_WEATHER_LIVE", "0")      # no Open-Meteo request from the test suite
os.environ.setdefault("COKE_ENERGY_TODAY", "2026-10-15")   # fixed plan day: its Open-Meteo moisture is in the shipped file
os.environ.setdefault("BF_TWIN_DB_URL", "off")             # no job archive into a DATABASE_URL of the developer's shell
# A test worker schedules no job that reads plant data (forecasts, twin, optimiser, calibration, STRATA-Learn, shift
# reports) and catches up nothing at start; the schedule tests enable both explicitly.
os.environ.setdefault("BF_WORKER_SKIP_SCHEDULES", "forecast.,twin.,optimiser.,calibration.,learn.,report.")
os.environ.setdefault("BF_WORKER_CATCH_UP", "0")
# STRATA-Flow network solves one after the other; the parallel tests start their solver processes themselves.
os.environ.setdefault("BF_STRATA_SOLVERS", "0")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")  # utils.settings needs a Qdrant URL; no test connects
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")  # utils.settings needs an LLM key; no test calls one


def _prefer_local_furnace_data_package() -> None:
    """Prepend local source paths for deterministic imports; append the repository root for ``tests.*`` helpers."""
    repo_root = Path(__file__).resolve().parent.parent
    for source_path in [repo_root / "src", repo_root / "furnace_data"]:
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
    if str(repo_root) not in sys.path:  # `pytest` (unlike `python -m pytest`) does not put the working directory first
        sys.path.append(str(repo_root))


_prefer_local_furnace_data_package()

# CI sets BF_TEST_REQUIRE_SERVICES=1: there a test whose service (PostgreSQL, Redis, Caddy) is missing fails instead of
# being skipped, so a broken service container cannot turn the integration tests into silent skips.
REQUIRE_SERVICES = os.environ.get("BF_TEST_REQUIRE_SERVICES") == "1"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_service(available, reason): skip without the service; fail instead when BF_TEST_REQUIRE_SERVICES=1",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    for marker in item.iter_markers(name="requires_service"):
        available, reason = marker.args
        if not available:
            if REQUIRE_SERVICES:
                pytest.fail(f"{reason} (BF_TEST_REQUIRE_SERVICES=1)", pytrace=False)
            pytest.skip(reason)
