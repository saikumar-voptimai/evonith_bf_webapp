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


def test_schematic_outline_follows_the_configured_profile() -> None:
    data = exporter.schematic_fixture()
    xs = sorted({round(x, 1) for x, _ in data["outline_points"]})
    # 16 vertices: the base point and seven profile points on each side, mirrored about x = 280 (CX).
    assert len(data["outline_points"]) == 16
    assert all(round(560 - x, 1) in xs for x in xs)
