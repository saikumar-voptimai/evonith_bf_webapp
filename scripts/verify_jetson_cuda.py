#!/usr/bin/env python3
"""Verify Jetson CUDA, PyTorch, and XGBoost GPU execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


def _enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _command(args: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    output = (result.stdout or result.stderr).strip()
    return {"ok": result.returncode == 0, "output": output[-1000:]}


def _torch_status() -> dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        return {
            "installed": True,
            "version": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "cuda_available": available,
            "device_name": torch.cuda.get_device_name(0) if available else None,
        }
    except Exception as exc:
        return {"installed": False, "cuda_available": False, "error": str(exc)}


def _xgboost_status() -> dict[str, Any]:
    try:
        import xgboost as xgb

        build_info = xgb.build_info()
        dmatrix = xgb.DMatrix([[1.0, 2.0], [2.0, 1.0]], label=[1.0, 0.0])
        booster = xgb.train(
            {"device": "cuda:0", "tree_method": "hist", "max_depth": 1},
            dmatrix,
            num_boost_round=1,
        )
        configured = str(booster.save_config())
        used_cuda = '"device":"cuda:0"' in configured
        return {
            "installed": True,
            "version": xgb.__version__,
            "cuda_built": _enabled(build_info.get("USE_CUDA")),
            "compiled_cuda": build_info.get("CUDA_VERSION"),
            "cuda_execution": used_cuda,
        }
    except Exception as exc:
        return {
            "installed": False,
            "cuda_built": False,
            "cuda_execution": False,
            "error": str(exc),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-torch", action="store_true")
    parser.add_argument("--require-xgboost", action="store_true")
    args = parser.parse_args(argv)

    cuda_paths = (Path("/dev/nvhost-gpu"), Path("/dev/nvidia0"))
    report = {
        "architecture": platform.machine(),
        "jetson_linux": Path("/etc/nv_tegra_release").exists(),
        "cuda_device_present": any(path.exists() for path in cuda_paths),
        "cuda_device_accessible": any(
            path.exists() and os.access(path, os.R_OK | os.W_OK)
            for path in cuda_paths
        ),
        "nvcc": _command(["/usr/local/cuda/bin/nvcc", "--version"]),
        "nvidia_smi": _command(["nvidia-smi"]),
        "torch": _torch_status(),
        "xgboost": _xgboost_status(),
    }
    failures: list[str] = []
    if report["architecture"] != "aarch64" or not report["jetson_linux"]:
        failures.append("not running on Jetson aarch64")
    if not report["cuda_device_present"]:
        failures.append("CUDA device node is unavailable")
    elif not report["cuda_device_accessible"]:
        failures.append("CUDA device node is not accessible to this service user")
    if args.require_torch and not report["torch"].get("cuda_available"):
        failures.append("PyTorch CUDA execution is unavailable")
    if args.require_xgboost and not report["xgboost"].get("cuda_execution"):
        failures.append("XGBoost CUDA execution is unavailable")
    report["status"] = "ok" if not failures else "failed"
    report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
