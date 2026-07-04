#!/usr/bin/env python
"""Import the backend app with edge-like optional features disabled."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import types


FORBIDDEN_STARTUP_MODULES = {
    "anthropic",
    "openai",
    "qdrant_client",
    "sentence_transformers",
    "streamlit",
    "torch",
}


def _configure_env(root: Path) -> None:
    os.environ.setdefault("EVONITH_RUNTIME_DIR", str(root / "runtime"))
    os.environ.setdefault("EVONITH_AUTH_SECRET_KEY", "dev-only-secret-change-me")
    os.environ.setdefault("EVONITH_RUNTIME_PROFILE", "edge")
    os.environ.setdefault("EVONITH_EDGE_MODE", "true")
    os.environ.setdefault("EVONITH_ENABLE_OPTIONAL_AI", "false")
    os.environ.setdefault("EVONITH_ENABLE_OPTIONAL_VECTOR", "false")
    os.environ.setdefault("EVONITH_ENABLE_OPTIONAL_LOCAL_LLM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    _configure_env(root)

    simulated = os.getenv("EVONITH_CHECK_SIMULATE_IMPORTED_MODULE", "").strip()
    if simulated:
        sys.modules[simulated] = types.ModuleType(simulated)

    sys.path.insert(0, str(root))
    try:
        from apps.backend_api.app.main import app
        from fastapi.testclient import TestClient

        if not app.title:
            raise RuntimeError("Backend app title is empty.")
        schema = app.openapi()
        if not schema.get("openapi"):
            raise RuntimeError("OpenAPI schema could not be generated.")
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/health")
        if response.status_code != 200:
            raise RuntimeError(f"Health endpoint returned {response.status_code}.")
    except Exception as exc:
        print("FAIL backend-minimal startup check")
        print(str(exc))
        return 1

    loaded = sorted(module for module in FORBIDDEN_STARTUP_MODULES if module in sys.modules)
    if loaded:
        print("FAIL backend-minimal startup check")
        print(f"Forbidden modules loaded during backend startup: {', '.join(loaded)}")
        return 1

    print("PASS backend-minimal startup check")
    print(f"app_title={app.title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
