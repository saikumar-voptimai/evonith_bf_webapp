"""One-shot diagnostic for the ML dataset fetch pipeline.

Calls every fetch step in isolation for a given date window, reports
row counts, NaN coverage, and the rename_dict join between sources.
Output is a markdown report on stdout (plus written to
``scripts/diagnose_fetch_pipeline.report.md``).

Run from the project root:

    python scripts/diagnose_fetch_pipeline.py
"""

from __future__ import annotations

import io
import sys
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

# Force UTF-8 stdout so arrow glyphs render on Windows consoles (cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "furnace_data"))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    # Fallback: parse .env manually if python-dotenv is absent.
    import os
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

import pandas as pd

from furnace_data.config import load_config
from furnace_data.dataset.fetcher import DatasetFetcher
from furnace_data.dataset.service import DatasetService
from furnace_data.neon_db.offline import fetch_offline_data


REPORT_PATH = REPO_ROOT / "scripts" / "diagnose_fetch_pipeline.report.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summarize_df(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {"rows": 0, "cols": 0, "time_range": "(empty)", "top_nan": []}
    info = {
        "rows": len(df),
        "cols": len(df.columns),
        "time_range": (
            f"{df.index.min()} → {df.index.max()}"
            if isinstance(df.index, pd.DatetimeIndex)
            else "(non-datetime index)"
        ),
        "top_nan": [],
    }
    nan_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    info["top_nan"] = [
        (col, round(float(pct), 1)) for col, pct in nan_pct.head(10).items() if pct > 0
    ]
    info["all_nan_cols"] = sorted(c for c in df.columns if df[c].isna().all())
    info["non_nan_cols"] = sorted(c for c in df.columns if df[c].notna().any())
    return info


def _print_step(name: str, df: pd.DataFrame, err: str | None = None) -> None:
    print(f"\n### {name}")
    if err is not None:
        print(f"\n> **FAILED** with exception:\n```\n{err}\n```")
        return
    s = _summarize_df(df)
    print(f"- rows: **{s['rows']}**, cols: **{s['cols']}**")
    print(f"- time range: {s['time_range']}")
    if s["rows"] == 0:
        return
    print(f"- columns with any data: **{len(s['non_nan_cols'])}** / {s['cols']}")
    if s["all_nan_cols"]:
        print(
            f"- columns all-NaN: **{len(s['all_nan_cols'])}** — "
            f"{', '.join(s['all_nan_cols'][:8])}"
            + (" …" if len(s["all_nan_cols"]) > 8 else "")
        )
    if s["top_nan"]:
        print("- top NaN% columns:")
        for col, pct in s["top_nan"][:6]:
            print(f"  - `{col}` — {pct}%")


@dataclass
class StepResult:
    name: str
    df: pd.DataFrame
    err: str | None = None


def _safe_call(name: str, fn: Callable[[], pd.DataFrame]) -> StepResult:
    try:
        df = fn()
        return StepResult(name=name, df=df if df is not None else pd.DataFrame())
    except Exception:  # noqa: BLE001
        return StepResult(name=name, df=pd.DataFrame(), err=traceback.format_exc())


# ---------------------------------------------------------------------------
# Diagnostic per window
# ---------------------------------------------------------------------------


def diagnose_window(start: date, end: date, rm_mode: str = "charge") -> None:
    cutoff = DatasetService.cutoff_date
    where = (
        "pre-cutoff (Step 1 only)"
        if end <= cutoff
        else "post-cutoff (Steps 2-5)"
        if start > cutoff
        else "straddling cutoff"
    )
    print(f"\n## Window: {start} → {end}  ({where}; cutoff={cutoff})")

    service = DatasetService()
    fetcher = DatasetFetcher(service=service)

    # ----- per-step isolated calls
    steps: list[StepResult] = []

    if end > cutoff:
        steps.append(_safe_call("Step 2 — fetch_rm_data (RM + chemistry + strength)",
                                lambda: service.fetch_rm_data(start, end, mode=rm_mode)))
        steps.append(_safe_call("Step 3 — fetch_hotmetal_hourly (HM+Slag 1h interp)",
                                lambda: service.fetch_hotmetal_hourly(start, end)))
        steps.append(_safe_call("Step 4 — fetch_distribution_data (burden daily)",
                                lambda: service.fetch_distribution_data(start, end)))
        steps.append(_safe_call("Step 5  — fetch_online_process_params (Influx process_params, 1h)",
                                lambda: service.fetch_online_process_params(start, end)))
        steps.append(_safe_call("Step 5b — fetch_online_temperature_params (Influx temperature_profile, 1h)",
                                lambda: service.fetch_online_temperature_params(start, end)))

    if start <= cutoff:
        steps.append(_safe_call("Step 1 — fetch (historical_static_ml_dataset)",
                                lambda: service.fetch(start, min(end, cutoff))))

    for step in steps:
        _print_step(step.name, step.df, step.err)

    # ----- join overlap (post-cutoff only)
    if end > cutoff:
        post_start = max(start, cutoff)
        join = _safe_call(
            f"Combined `_build_post_cutoff_df` ({post_start} → {end})",
            lambda: fetcher._build_post_cutoff_df(post_start, end, rm_mode),  # noqa: SLF001
        )
        print(f"\n### {join.name}")
        if join.err:
            print(f"\n> **FAILED**:\n```\n{join.err}\n```")
        else:
            s = _summarize_df(join.df)
            print(f"- rows: **{s['rows']}**, cols: **{s['cols']}**")
            print(f"- columns with any data: **{len(s['non_nan_cols'])}** / {s['cols']}")
            if s["all_nan_cols"]:
                print(
                    f"- all-NaN columns: **{len(s['all_nan_cols'])}** — "
                    f"{', '.join(s['all_nan_cols'][:10])}"
                    + (" …" if len(s["all_nan_cols"]) > 10 else "")
                )

    # ----- target UI columns the user asked about specifically
    if end > cutoff and steps:
        # service.fetch_online_process_params and ..._temperature_params now
        # produce Neon-named columns (after rename via online_params/temperature_params).
        # We want to verify the new entries actually populate the right columns.
        watch_targets = {
            "top_bar": "TOPBAR",                       # body_dp_top
            "oxygen_flow_nm3hr": "OXYGENFLOWNM3/HR.",  # oxygen_flow
            "charges_per_hr": "CHARGES/HRS.",          # charges_per_hour
            "ftg_uptake_cat16_c": "FTG_UPTAKE_TEMP_A", # top_temp_1
            "ftg_uptake_bt12_c": "FTG_UPTAKE_TEMP_B",  # top_temp_2
            "ftg_uptake_ct08_c": "FTG_UPTAKE_TEMP_C",  # top_temp_3
            "ftg_uptake_dt04_c": "FTG_UPTAKE_TEMP_D",  # top_temp_4
            "ftg_uptake_avg_c": "FTG_UPTAKE_TEMP_AVG", # top_temp_avg
            "hearth_pad_a_c": "HEARTH_TEMP_A",         # temp_4373_a
            "hearth_pad_b_c": "HEARTH_TEMP_B",         # temp_4373_b
            "hearth_pad_c_c": "HEARTH_TEMP_C",         # temp_4373_c
            "hearth_pad_d_c": "HEARTH_TEMP_D",         # temp_4373_d
        }
        print("\n### Newly mapped columns — did they actually fill?")
        process_step = next((s for s in steps if "process_params" in s.name), None)
        temp_step = next((s for s in steps if "temperature_profile" in s.name), None)

        def _check_filled(df: pd.DataFrame, neon_col: str, ui_col: str) -> str:
            if df is None or df.empty:
                return "no rows"
            if neon_col not in df.columns:
                return "absent from fetch output"
            non_null = int(df[neon_col].notna().sum())
            total = len(df)
            pct = (100 * non_null / total) if total else 0
            return f"{non_null}/{total} non-null ({pct:.1f}%)"

        for neon, ui in watch_targets.items():
            owner = "process_params" if "uptake" in neon or neon in (
                "top_bar", "oxygen_flow_nm3hr", "charges_per_hr",
            ) else "temperature_profile"
            df = process_step.df if owner == "process_params" else temp_step.df
            status = _check_filled(df, neon, ui)
            print(f"- `{neon}` → `{ui}`  ({owner}): {status}")


# ---------------------------------------------------------------------------
# Rename audit — what's in rename_dict but never produced by any source
# ---------------------------------------------------------------------------


def rename_audit() -> None:
    cfg = load_config("setting_ds_dv.yml")
    rename_dict: dict = cfg["rename_dict"]
    online_params: dict = cfg["ml_dataset"]["online_params"]
    temperature_params: dict = cfg["ml_dataset"]["temperature_params"]

    influx_sourced = set(online_params.values()) | set(temperature_params.values())

    # Try a tiny inspection to see which rename_dict keys exist as columns in
    # the historical_static_ml_dataset table.
    try:
        sample = fetch_offline_data(
            table_name="offline_feed.historical_static_ml_dataset",
            time_range="last 1 day",
            query_type="raw",
        )
        neon_sourced = set(sample.columns) if sample is not None and not sample.empty else set()
    except Exception:
        neon_sourced = set(rename_dict.keys())  # fallback: assume the rename_dict keys all exist
        print("> ⚠️ Could not query historical_static_ml_dataset live; assuming all rename_dict keys exist.")

    no_source = []
    only_neon = []
    only_influx = []
    both = []
    for neon_name, ui_name in rename_dict.items():
        in_influx = neon_name in influx_sourced
        in_neon = neon_name in neon_sourced
        if in_influx and in_neon:
            both.append((neon_name, ui_name))
        elif in_influx:
            only_influx.append((neon_name, ui_name))
        elif in_neon:
            only_neon.append((neon_name, ui_name))
        else:
            no_source.append((neon_name, ui_name))

    print("\n## Rename-dict source audit")
    print(f"- rename_dict entries: **{len(rename_dict)}**")
    print(f"- sourced by both Neon AND Influx (overlapping): **{len(both)}**")
    print(f"- sourced ONLY from Neon historical (Step 1): **{len(only_neon)}**")
    print(f"- sourced ONLY from Influx (Steps 5/5b): **{len(only_influx)}**")
    print(f"- NO source identified: **{len(no_source)}**")

    if only_influx:
        print("\n### Influx-only columns (new post-cutoff fills)")
        for neon, ui in sorted(only_influx, key=lambda t: t[1]):
            print(f"- `{neon}` → `{ui}`")

    if no_source:
        print("\n### rename_dict entries with no Step-1/5 source")
        print("(These rename_dict targets are aspirational — they're in the cleaning schema but no step produces them.)")
        for neon, ui in sorted(no_source, key=lambda t: t[1]):
            print(f"- `{neon}` → `{ui}`")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        print("# ML Dataset Fetch-Pipeline Diagnostic")
        print()
        print("Generated by `scripts/diagnose_fetch_pipeline.py` after Part 0 yml updates.")
        print()
        print("Sample windows are chosen to probe (a) the pre-cutoff Step 1 path, "
              "(b) the straddling case, and (c) the post-cutoff Steps 2-5 with the "
              "newly-added Influx → Neon mappings.")
        print()

        windows = [
            (date(2025, 8, 1), date(2025, 8, 7)),     # pre-cutoff
            (date(2025, 12, 1), date(2025, 12, 12)),  # straddle
            (date(2026, 4, 1), date(2026, 4, 7)),     # post-cutoff
        ]
        for start, end in windows:
            try:
                diagnose_window(start, end, rm_mode="charge")
            except Exception:  # noqa: BLE001
                print(f"\n> ⚠️ Window {start}→{end} failed top-level:")
                print("```")
                print(traceback.format_exc())
                print("```")

        try:
            rename_audit()
        except Exception:  # noqa: BLE001
            print("\n> ⚠️ Rename audit failed:")
            print("```")
            print(traceback.format_exc())
            print("```")

    report = buffer.getvalue()
    print(report)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n[report saved to {REPORT_PATH.relative_to(REPO_ROOT)}]")


if __name__ == "__main__":
    main()
