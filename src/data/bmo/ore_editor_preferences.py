"""Persistence helpers for BMO ore editor defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PERSISTED_NUMERIC_COLUMNS = (
    "price_rs_per_mt",
    "min_share_pct",
    "max_share_pct",
)


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_ore_editor_preferences(editor_df: pd.DataFrame) -> dict[str, Any]:
    """Build a compact, disk-safe preference payload from the ore editor."""

    if editor_df.empty or "ore_id" not in editor_df.columns:
        return {"ore_editor": {"selected_ore_ids": [], "rows": {}}}

    selected_ids: list[str] = []
    rows: dict[str, dict[str, float]] = {}
    for _, row in editor_df.iterrows():
        ore_id = str(row.get("ore_id", "")).strip()
        if not ore_id:
            continue
        if bool(row.get("selected", False)):
            selected_ids.append(ore_id)
        saved_row: dict[str, float] = {}
        for column in PERSISTED_NUMERIC_COLUMNS:
            if column not in row:
                continue
            value = _float_or_none(row.get(column))
            if value is not None:
                saved_row[column] = value
        rows[ore_id] = saved_row

    return {"ore_editor": {"selected_ore_ids": selected_ids, "rows": rows}}


def apply_ore_editor_preferences(
    editor_df: pd.DataFrame, preferences: dict[str, Any]
) -> pd.DataFrame:
    """Apply saved ore editor preferences to a freshly built editor frame."""

    if editor_df.empty or "ore_id" not in editor_df.columns:
        return editor_df

    ore_editor = preferences.get("ore_editor", {}) if preferences else {}
    saved_rows = ore_editor.get("rows", {}) or {}
    selected_ids = {str(ore_id) for ore_id in ore_editor.get("selected_ore_ids", [])}
    out = editor_df.copy()

    if selected_ids and "selected" in out.columns:
        out["selected"] = out["ore_id"].astype(str).isin(selected_ids)

    for index, row in out.iterrows():
        ore_id = str(row.get("ore_id", "")).strip()
        saved = saved_rows.get(ore_id, {}) or {}
        for column in PERSISTED_NUMERIC_COLUMNS:
            if column in out.columns and column in saved:
                value = _float_or_none(saved[column])
                if value is not None:
                    out.at[index, column] = value
    return out


def load_ore_editor_preferences(path: str | Path) -> dict[str, Any]:
    """Load saved BMO ore editor preferences, returning empty prefs if absent."""

    pref_path = Path(path)
    if not pref_path.exists():
        return {}
    with open(pref_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data if isinstance(data, dict) else {}


def save_ore_editor_preferences(path: str | Path, editor_df: pd.DataFrame) -> Path:
    """Persist BMO ore editor preferences and return the written path."""

    pref_path = Path(path)
    pref_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_ore_editor_preferences(editor_df)
    with open(pref_path, "w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    return pref_path
