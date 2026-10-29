"""Parity fixtures of the Svelte chart kit (#275) are current: they equal the Python models' output.

Regenerate with ``python scripts/export_chart_fixtures.py``; the Vitest suite compares the Svelte drawings with them.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_chart_fixtures", ROOT / "scripts" / "export_chart_fixtures.py")
assert spec is not None and spec.loader is not None
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)

HINT = "Regenerate with: python scripts/export_chart_fixtures.py"


@pytest.mark.parametrize(
    ("path", "build"),
    [
        (exporter.RIST_OUT, exporter.rist_fixture),
        (exporter.SCHEMATIC_OUT, exporter.schematic_fixture),
        (exporter.STATUS_OUT, exporter.status_fixture),
        (exporter.REPORT_OUT, exporter.report_fixture),
    ],
    ids=["rist", "schematic", "status", "report"],
)
def test_fixture_is_current(path: Path, build) -> None:
    committed = json.loads(path.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(build())), HINT


def _differences(a: object, b: object, where: str = "") -> list[str]:
    """Paths where two JSON documents differ; numbers within a relative 1e-6 (an LP optimum may move in its last
    digits between platforms) count as equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{where}: keys {sorted(a.keys() ^ b.keys())}"]
        return [d for k in a for d in _differences(a[k], b[k], f"{where}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{where}: {len(a)} against {len(b)} items"]
        return [d for i, (x, y) in enumerate(zip(a, b, strict=True)) for d in _differences(x, y, f"{where}[{i}]")]
    numbers = (int, float)
    if isinstance(a, numbers) and isinstance(b, numbers) and not isinstance(a, bool) and not isinstance(b, bool):
        return [] if abs(a - b) <= 1e-6 * max(1.0, abs(b)) else [f"{where}: {a} against {b}"]
    return [] if a == b else [f"{where}: {a!r} against {b!r}"]


def test_optimiser_fixture_is_current() -> None:
    committed = json.loads(exporter.OPTIMISER_OUT.read_text(encoding="utf-8"))
    fresh = json.loads(json.dumps(exporter.optimiser_fixture()))
    found = _differences(committed, fresh)
    assert not found, f"{HINT}\n" + "\n".join(found[:20])


def test_schematic_outline_follows_the_configured_profile() -> None:
    data = exporter.schematic_fixture()
    xs = sorted({round(x, 1) for x, _ in data["outline_points"]})
    # 16 vertices: the base point and seven profile points on each side, mirrored about x = 280 (CX).
    assert len(data["outline_points"]) == 16
    assert all(round(560 - x, 1) in xs for x in xs)
