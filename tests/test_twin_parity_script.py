"""``scripts/twin_parity.py`` (SV-1.1, #281): the plant-data parity check, run here on the synthetic sources.

The script loads the sources through ``services.coke_energy`` and ``services.twin.load_frames``; the test replaces
both with the synthetic frames of ``tests/twin_data.py`` and their pipeline result.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from services import coke_energy as ce
from services import twin
from tests.twin_data import synthetic_frames
from utils.coke_energy.pipeline import PipelineSettings, run_pipeline

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("twin_parity", ROOT / "scripts" / "twin_parity.py")
assert spec is not None and spec.loader is not None
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)

CFG = {"history_start": "2026-04-01", "predict_from": "2026-04-01", "rolling_window_days": 14, "rolling_min_shifts": 10}


@pytest.fixture(scope="module")
def sources():
    fd, m10, hm, dpr = synthetic_frames(start="2026-05-01", days=18)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_pipeline(fd, m10, hm, dpr, PipelineSettings.from_config(CFG))
    return twin.TwinFrames(fd, m10, hm, dpr, ()), result


def _patch(monkeypatch, frames, result) -> None:
    monkeypatch.setattr(ce, "config", lambda: CFG)
    monkeypatch.setattr(ce, "data_keys", lambda cfg, nonce: "keys")
    monkeypatch.setattr(ce, "page_data", lambda keys, settings: SimpleNamespace(result=result))
    monkeypatch.setattr(twin, "load_frames", lambda keys: frames)


def test_windows_of_the_last_days_equal_the_pipeline(sources, monkeypatch, tmp_path, capsys) -> None:
    frames, result = sources
    _patch(monkeypatch, frames, result)
    out = tmp_path / "parity.json"
    assert parity.main(["--days", "3", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "9 shifts from 2026-05-16 to 2026-05-18" in text and "FAIL" not in text
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert len(rows) == 9 and {r["status"] for r in rows} == {"complete"}


def test_a_difference_beyond_the_tolerance_fails(sources, monkeypatch, capsys) -> None:
    frames, result = sources
    shifts = result.shifts.copy()
    shifts.loc["2026-05-18_A", "K_energy"] += 0.2  # 0.2 kg/tHM > 0.1
    _patch(monkeypatch, frames, replace(result, shifts=shifts))
    assert parity.main(["--days", "1"]) == 1
    assert "FAIL 2026-05-18_A: k_forge differs by 0.200000 (tolerance 0.1)" in capsys.readouterr().out
