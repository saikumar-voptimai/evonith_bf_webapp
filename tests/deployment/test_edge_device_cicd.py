from __future__ import annotations

import os
from pathlib import Path
import subprocess

from scripts.verify_edge_device import detect_device


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_detects_jetson_arm64(tmp_path: Path) -> None:
    release = tmp_path / "nv_tegra_release"
    release.write_text("# R36", encoding="utf-8")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:        8040000 kB\n", encoding="utf-8")

    result = detect_device(
        architecture="aarch64",
        jetson_release=release,
        device_model=tmp_path / "missing-model",
        meminfo=meminfo,
    )

    assert result["status"] == "ok"
    assert result["device_family"] == "jetson"
    assert result["recommended_dependency_group"] == "edge"


def test_detects_64_bit_raspberry_pi(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.write_text("Raspberry Pi 5 Model B Rev 1.0\x00", encoding="utf-8")

    result = detect_device(
        architecture="aarch64",
        jetson_release=tmp_path / "missing-release",
        device_model=model,
        meminfo=tmp_path / "missing-meminfo",
    )

    assert result["status"] == "ok"
    assert result["device_family"] == "raspberry-pi"
    assert result["device_model"] == "Raspberry Pi 5 Model B Rev 1.0"


def test_rejects_non_arm64_edge_host(tmp_path: Path) -> None:
    result = detect_device(
        architecture="armv7l",
        jetson_release=tmp_path / "missing-release",
        device_model=tmp_path / "missing-model",
        meminfo=tmp_path / "missing-meminfo",
    )

    assert result["status"] == "fail"
    assert result["device_family"] == "unsupported"
    assert "64-bit ARM Linux is required" in result["failures"][0]


def test_deploy_script_has_safe_dry_run() -> None:
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "deploy_edge_release.sh"),
            "--ref",
            "0123456789abcdef",
            "--device",
            "raspberry-pi",
            "--repo-dir",
            "/not-used-in-dry-run",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[dry-run]" in result.stdout
    assert "branch=release" in result.stdout
    assert "refs/remotes/origin/release" in result.stdout
    assert "systemctl restart evonith-backend" in result.stdout
    assert "--group edge --locked" in result.stdout


def test_host_bootstrap_dry_run_is_non_destructive() -> None:
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "bootstrap_edge_host.sh"),
            "--device",
            "raspberry-pi",
            "--repo-dir",
            str(REPO_ROOT),
            "--deploy-owner",
            os.environ.get("USER", "root"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[dry-run]" in result.stdout
    assert "systemctl daemon-reload" in result.stdout
    assert "Host bootstrap complete for raspberry-pi" in result.stdout


def test_device_templates_and_workflow_are_safe() -> None:
    jetson = (REPO_ROOT / "infra" / "env" / "edge.jetson.env.example").read_text()
    raspberry_pi = (
        REPO_ROOT / "infra" / "env" / "edge.raspberry-pi.env.example"
    ).read_text()
    workflow = (REPO_ROOT / ".github" / "workflows" / "edge-ci-cd.yml").read_text()
    sudoers = (
        REPO_ROOT / "infra" / "sudoers" / "evonith-edge-deploy.example"
    ).read_text()

    assert "EVONITH_EDGE_DEVICE_TYPE=jetson" in jetson
    assert "EVONITH_ML_DEVICE=auto" in jetson
    assert "EVONITH_EDGE_DEVICE_TYPE=raspberry-pi" in raspberry_pi
    assert "EVONITH_ML_DEVICE=cpu" in raspberry_pi
    assert "EVONITH_CUDA_REQUIRED=false" in raspberry_pi
    assert "EVONITH_UVICORN_HOST=127.0.0.1" in jetson
    assert "EVONITH_UVICORN_HOST=127.0.0.1" in raspberry_pi
    assert "evonith-jetson" in workflow
    assert "evonith-pi" in workflow
    assert "needs: test" in workflow
    assert "persist-credentials: false" in workflow
    assert "EVONITH_DEPLOY_TARGET" in workflow
    assert "refs/heads/release" in workflow
    assert workflow.count("--branch release") == 2
    assert "default: pi" in workflow
    assert "DEPLOY_JETSON_ENABLED" not in workflow
    assert "DEPLOY_PI_ENABLED" not in workflow
    assert "refs/heads/dev-v01'" not in workflow
    assert "systemctl restart evonith-backend" in sudoers
    assert "ALL=(ALL)" not in sudoers
