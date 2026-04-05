"""Prompt builders for AI Copilot.

All static analysis text lives in ``src/assets/data/copilot_analysis/``.
The functions here read those files at call time so the page always serves
the latest version without a restart.

Files:
  BURDEN_UNITCOST.md      — burden distribution + unit cost findings
  ANOMALY_SENSOR_DESC.md  — furnace sensor/level descriptions
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.copilot.data import df_packet

_ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "assets" / "data" / "copilot_analysis"

# ── System prompts (short persona strings, stable — no need to externalise) ──

ANOMALY_SYSTEM = (
    "You are a careful anomaly detector for blast furnace thermal and gas behavior."
)

BURDEN_SYSTEM = (
    "You are a precise, senior blast furnace burden advisor. Be concise, numeric, and actionable."
)


# ── File loaders ─────────────────────────────────────────────────────────────

def _read_analysis_file(name: str) -> str:
    """Read a file from the copilot_analysis directory; return empty string on failure."""
    path = _ANALYSIS_DIR / name
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return f"_(Analysis file `{name}` not found — update `src/assets/data/copilot_analysis/`)_"


def load_burden_findings() -> str:
    """Return the full BURDEN_UNITCOST.md content."""
    return _read_analysis_file("BURDEN_UNITCOST.md")


def load_sensor_desc() -> str:
    """Return the ANOMALY_SENSOR_DESC.md content."""
    return _read_analysis_file("ANOMALY_SENSOR_DESC.md")


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_burden_prompt(findings: str) -> str:
    """Wrap the pre-loaded findings text in a task instruction for the LLM.

    The LLM is told the analysis has already been done; its job is to narrate
    the findings clearly — no hallucination, no repeating maths details.
    """
    return (
        "I have previously run a regression analysis on burden distribution and unit cost. "
        "The findings are below. Present them clearly and concisely in Markdown. "
        "Do not hallucinate. Do not repeat mathematical details (betas, rho values). "
        "Do not repeat yourself.\n\n"
        f"{findings}"
    )


def build_anomaly_prompt(
    recent_df: pd.DataFrame,
    past_df: pd.DataFrame,
    sensor_desc: str,
    notes: str = "",
) -> str:
    """Build the anomaly analysis prompt from live DataFrames + sensor description.

    Parameters
    ----------
    recent_df:   Last 8 hours of telemetry (15-min averages).
    past_df:     8–24 hours ago (15-min averages).
    sensor_desc: Content of ANOMALY_SENSOR_DESC.md — furnace zone/sensor layout.
    notes:       Optional operator notes entered in the UI.
    """
    pkt_recent = df_packet(recent_df) if not recent_df.empty else "_No timeseries to show_"
    pkt_past   = df_packet(past_df)   if not past_df.empty   else "_No past timeseries to show_"

    return f"""\
You are an anomaly spotter and shift summariser helping the operator taking over the next shift.

## Furnace sensor layout
{sensor_desc}

## Task
Review the **last 8 hours** for:
- Furnace profile temperature spikes
- Heatload spikes or quadrant asymmetry
- ΔT excursions
- Gas/pressure instabilities (permeability fall, ΔP rise, top-pressure swings)

## Recent 8 hours (averaged to 15 min)
{pkt_recent}

## Past 8–24 hours (averaged to 15 min)
{pkt_past}

## Operator notes
{notes if notes else "None"}

## Output — up to 200 words only
- Significant changes between the two windows (brief).
- Key observations for the incoming operator: blowdowns, startups, shutdowns, fuel rate trend, furnace stability.
- Alerts (issue + severity).
- Likely causes mapped to controllables (HB temp/volume/pressure, O₂, steam, PCI, burden quality).

NOTE: Stick strictly to the provided data. Do not ask questions or expect further input.\
"""
