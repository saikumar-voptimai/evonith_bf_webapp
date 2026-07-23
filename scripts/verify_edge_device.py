#!/usr/bin/env python3
"""Verify that a host is suitable for an Evonith ARM64 edge deployment."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import sys
from typing import Any


SUPPORTED_ARCHITECTURES = {"aarch64", "arm64"}
DEVICE_ALIASES = {
    "auto": "auto",
    "generic": "generic-arm64",
    "generic-arm64": "generic-arm64",
    "jetson": "jetson",
    "jetson-orin-nano": "jetson",
    "raspberry-pi": "raspberry-pi",
    "raspberry-pi-4": "raspberry-pi",
    "raspberry-pi-5": "raspberry-pi",
    "pi": "raspberry-pi",
}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").replace("\x00", "").strip()
    except OSError:
        return ""


def _memory_mb(meminfo: Path = Path("/proc/meminfo")) -> int | None:
    for line in _read_text(meminfo).splitlines():
        if line.startswith("MemTotal:"):
            try:
                return int(line.split()[1]) // 1024
            except (IndexError, ValueError):
                return None
    return None


def detect_device(
    *,
    architecture: str | None = None,
    jetson_release: Path = Path("/etc/nv_tegra_release"),
    device_model: Path = Path("/proc/device-tree/model"),
    meminfo: Path = Path("/proc/meminfo"),
) -> dict[str, Any]:
    """Return dependency-free host capability information."""
    architecture = (architecture or platform.machine()).strip().lower()
    model = _read_text(device_model)
    jetson = jetson_release.is_file()
    raspberry_pi = "raspberry pi" in model.lower()

    if jetson:
        device_family = "jetson"
    elif raspberry_pi:
        device_family = "raspberry-pi"
    elif architecture in SUPPORTED_ARCHITECTURES:
        device_family = "generic-arm64"
    else:
        device_family = "unsupported"

    memory_mb = _memory_mb(meminfo)
    warnings: list[str] = []
    failures: list[str] = []
    if architecture not in SUPPORTED_ARCHITECTURES:
        failures.append(
            "64-bit ARM Linux is required; install a 64-bit OS that reports aarch64/arm64"
        )
    if memory_mb is not None and memory_mb < 3500:
        warnings.append("less than 4 GB RAM detected; 8 GB is recommended")

    return {
        "status": "fail" if failures else ("warn" if warnings else "ok"),
        "device_family": device_family,
        "device_model": model or None,
        "architecture": architecture,
        "cpu_count": os.cpu_count(),
        "memory_mb": memory_mb,
        "jetson_linux": jetson,
        "cuda_device_present": any(
            path.exists() for path in (Path("/dev/nvhost-gpu"), Path("/dev/nvidia0"))
        ),
        "recommended_dependency_group": "edge",
        "warnings": warnings,
        "failures": failures,
    }


def _expected_family(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized not in DEVICE_ALIASES:
        choices = ", ".join(sorted(DEVICE_ALIASES))
        raise ValueError(f"unsupported expected device {value!r}; choose one of: {choices}")
    return DEVICE_ALIASES[normalized]


def verify(expected: str = "auto") -> dict[str, Any]:
    result = detect_device()
    expected_family = _expected_family(expected)
    result["expected_device_family"] = expected_family
    if expected_family != "auto" and result["device_family"] != expected_family:
        result["failures"].append(
            f"expected {expected_family}, detected {result['device_family']}"
        )
        result["status"] = "fail"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect",
        default=os.getenv("EVONITH_EDGE_DEVICE_TYPE", "auto"),
        help="Expected device family (auto, jetson, raspberry-pi, or generic-arm64)",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)

    try:
        result = verify(args.expect)
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"{result['status'].upper()} edge_device: {result['device_family']}")
        print(f"architecture: {result['architecture']}")
        print(f"model: {result['device_model'] or 'unknown'}")
        print(f"memory_mb: {result['memory_mb'] if result['memory_mb'] is not None else 'unknown'}")
        for warning in result["warnings"]:
            print(f"WARN: {warning}")
        for failure in result["failures"]:
            print(f"FAIL: {failure}")
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
