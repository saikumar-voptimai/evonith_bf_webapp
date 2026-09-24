"""Snapshot panel for the Blend Mix Optimiser, and the loader for TestBMO.

``render_snapshot_panel`` sits at the foot of the optimiser page. It is called
with ``prefix="bmo_"``; inside TestBMO the same call arrives as ``"testbmo_"``
because the sandbox renames key-shaped strings in the page source, so the one
line on the page gives both pages a working panel.

Every widget here uses a ``<prefix>ui_`` key. Those are excluded from capture and
from restore: they are this panel's controls, not optimiser state.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from utils.bmo.sandbox import SANDBOX_PREFIX, UI_SUFFIX
from utils.bmo.snapshot import (
    SOURCE_BY_PREFIX,
    capture,
    restorable_state,
    validate,
)
from utils.bmo.snapshot_report import build_docx
from utils.bmo.snapshot_store import delete, list_snapshots, load, save

log = logging.getLogger(__name__)

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@st.cache_data(show_spinner=False, max_entries=16)
def _docx_bytes(snapshot_id: str, path: str, mtime_ns: int) -> bytes:
    """The report for one snapshot. Cached on the file's mtime - rebuilt only if
    the snapshot changes, not on every rerun of a long page."""

    return build_docx(load(snapshot_id))


def _mtime(path: str) -> int:
    from pathlib import Path

    try:
        return Path(path).stat().st_mtime_ns
    except OSError:
        return 0


def _table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    def blend(shares: Mapping[str, float]) -> str:
        return ", ".join(f"{k} {v:.0f}%" for k, v in list(shares.items())[:4])

    return pd.DataFrame([
        {
            "Taken at": r["created_at"].replace("T", " ")[:19],
            "Page": r["source"],
            "Label": r["label"],
            "Blend": blend(r["summary"].get("blend_pct") or {}),
            "Fuel rate": r["summary"].get("fuel_rate_kg_thm"),
            "Coke rate": r["summary"].get("coke_rate_kg_thm"),
            "Slag rate": r["summary"].get("slag_rate_kg_thm"),
            "B2": r["summary"].get("basicity_b2"),
            "T-basicity": r["summary"].get("t_basicity"),
            "Production": r["summary"].get("production_mt"),
            "Total cost": r["summary"].get("total_cost_rs_thm"),
            "Result": r["summary"].get("result") or "inputs only",
        }
        for r in rows
    ])


_TABLE_CONFIG = {
    "Fuel rate": st.column_config.NumberColumn("Fuel rate (kg/THM)", format="%.1f"),
    "Coke rate": st.column_config.NumberColumn("Coke rate (kg/THM)", format="%.1f"),
    "Slag rate": st.column_config.NumberColumn("Slag rate (kg/THM)", format="%.1f"),
    "B2": st.column_config.NumberColumn("B2", format="%.3f"),
    "T-basicity": st.column_config.NumberColumn("T-basicity", format="%.3f"),
    "Production": st.column_config.NumberColumn("Production (MT)", format="%.0f"),
    "Total cost": st.column_config.NumberColumn("Total cost (Rs/THM)", format="%.0f"),
}


def render_snapshot_panel(
    prefix: str = "bmo_", page_vars: Mapping[str, Any] | None = None
) -> None:
    """Take Snapshot, the snapshot table, and the report download.

    Args:
         - prefix: str - State namespace of the page calling this.
         - page_vars: Mapping | None - The calling page's globals, so the snapshot
           records the input tables the page ACTUALLY ran with.
    """

    ui = f"{prefix}{UI_SUFFIX}"
    st.divider()
    st.markdown("### 📸 Snapshots")
    st.caption(
        "A snapshot records every input on this page, the optimiser results and "
        "any commentary, as a timestamped JSON file. Download it as a report, or "
        "load it into **TestBMO** to replay and change it."
    )

    label_col, button_col = st.columns([3, 1], vertical_alignment="bottom")
    label = label_col.text_input(
        "Label (optional)", key=f"{ui}label",
        placeholder="e.g. Shift B — trial with 12% pellet",
    )
    if button_col.button("📸 Take Snapshot", type="primary", width="stretch",
                         key=f"{ui}take"):
        try:
            snapshot = capture(st.session_state, prefix=prefix,
                               page_vars=page_vars or {}, label=label)
            path = save(snapshot)
            st.success(f"Snapshot **{snapshot['id']}** saved — {path.name}")
            if not snapshot["results"]:
                st.info("No optimiser result was on the page, so this snapshot "
                        "holds inputs only. Run LP or DE first for a full report.")
        except Exception as exc:  # noqa: BLE001 - never take the page down
            log.exception("Snapshot failed")
            st.error(f"Could not take the snapshot: {exc}")

    rows, broken = list_snapshots()
    if broken:
        st.caption(f"⚠️ {len(broken)} snapshot file(s) could not be read: "
                   + ", ".join(broken[:5]))
    if not rows:
        st.info("No snapshots yet.")
        return

    st.dataframe(_table(rows), hide_index=True, width="stretch",
                 column_config=_TABLE_CONFIG)

    by_id = {r["id"]: r for r in rows}
    choice = st.selectbox(
        "Snapshot", list(by_id), key=f"{ui}choice",
        format_func=lambda sid: (
            f"{by_id[sid]['created_at'].replace('T', ' ')[:19]} · "
            f"{by_id[sid]['source']}"
            + (f" · {by_id[sid]['label']}" if by_id[sid]["label"] else "")
        ),
    )
    row = by_id[choice]
    report_col, json_col, delete_col = st.columns(3)
    try:
        report = _docx_bytes(choice, row["path"], _mtime(row["path"]))
        report_col.download_button(
            "⬇️ Download Snapshot (.docx)", data=report,
            file_name=f"BMO_snapshot_{choice}.docx", mime=DOCX_MIME,
            width="stretch", key=f"{ui}download_docx",
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Report build failed")
        report_col.error(f"Report could not be built: {exc}")
    try:
        with open(row["path"], "rb") as handle:
            json_col.download_button(
                "⬇️ Download JSON", data=handle.read(),
                file_name=f"BMO_snapshot_{choice}.json", mime="application/json",
                width="stretch", key=f"{ui}download_json",
            )
    except OSError as exc:
        json_col.error(str(exc))
    if delete_col.button("🗑️ Delete", width="stretch", key=f"{ui}delete"):
        delete(choice)
        st.rerun()


# --- TestBMO ---------------------------------------------------------------------


def _apply_to_sandbox(snapshot: Mapping[str, Any]) -> None:
    """Replace the sandbox's state with this snapshot's inputs, then rerun.

    Everything in the sandbox namespace is cleared first - inputs, results,
    cached flags - so nothing from the previous run leaks into the new one. Only
    the loader's own ``ui_`` widgets survive.
    """

    writes = restorable_state(snapshot, prefix=SANDBOX_PREFIX)
    keep = f"{SANDBOX_PREFIX}{UI_SUFFIX}"
    for key in [k for k in list(st.session_state.keys())
                if str(k).startswith(SANDBOX_PREFIX) and not str(k).startswith(keep)]:
        del st.session_state[key]
    # The JSON editor must re-read the new document, not keep the old text.
    st.session_state.pop(f"{keep}json_editor", None)
    for key, value in writes.items():
        st.session_state[key] = value
    st.session_state[f"{keep}loaded"] = {
        "id": snapshot.get("id") or "(unsaved)",
        "created_at": snapshot.get("created_at", ""),
        "source": snapshot.get("source", ""),
        "label": snapshot.get("label", ""),
        "summary": snapshot.get("summary") or {},
        "restored": len(writes),
        "not_restorable": list(snapshot.get("not_restorable") or []),
    }
    st.session_state[f"{keep}loaded_inputs"] = snapshot.get("inputs") or {}
    st.rerun()


def _reset_sandbox() -> None:
    keep = f"{SANDBOX_PREFIX}{UI_SUFFIX}"
    for key in [k for k in list(st.session_state.keys())
                if str(k).startswith(SANDBOX_PREFIX)]:
        if str(key) in {f"{keep}loaded", f"{keep}loaded_inputs", f"{keep}json_editor"} \
                or not str(key).startswith(keep):
            del st.session_state[key]
    st.rerun()


def render_sandbox_loader() -> None:
    """Top of TestBMO: what is loaded, and the three ways to load something else."""

    ui = f"{SANDBOX_PREFIX}{UI_SUFFIX}"
    loaded = st.session_state.get(f"{ui}loaded")

    st.warning(
        "🧪 **TestBMO — sandbox.** A separate copy of the Blend Mix Optimiser with "
        "its own state. Load any snapshot or JSON, change anything, run it. "
        "Nothing here touches the live optimiser page."
    )
    if loaded:
        s = loaded["summary"]
        origin = (
            f"taken {loaded['created_at'].replace('T', ' ')[:19]} on {loaded['source']}"
            if loaded["created_at"] else f"from {loaded['source'] or 'JSON'}"
        )
        st.info(
            f"Loaded **{loaded['id']}** — {origin}"
            + (f" · *{loaded['label']}*" if loaded["label"] else "")
            + f". {loaded['restored']} inputs restored."
            + (f" Recorded result ({s.get('result')}): total cost "
               f"{s.get('total_cost_rs_thm') or 0:,.0f} Rs/THM, coke "
               f"{s.get('coke_rate_kg_thm') or 0:,.1f} kg/THM, slag "
               f"{s.get('slag_rate_kg_thm') or 0:,.1f} kg/THM — run the optimiser "
               "below to compare." if s.get("result") else "")
        )

    with st.expander("📂 Load a state into the sandbox", expanded=not loaded):
        saved_tab, upload_tab, edit_tab = st.tabs(
            ["Saved snapshots", "Upload JSON", "Edit JSON"]
        )

        with saved_tab:
            rows, _broken = list_snapshots()
            if not rows:
                st.caption("No snapshots saved yet. Take one on the Blend Mix "
                           "Optimiser page.")
            else:
                by_id = {r["id"]: r for r in rows}
                choice = st.selectbox(
                    "Snapshot", list(by_id), key=f"{ui}saved_choice",
                    format_func=lambda sid: (
                        f"{by_id[sid]['created_at'].replace('T', ' ')[:19]} · "
                        f"{by_id[sid]['source']}"
                        + (f" · {by_id[sid]['label']}" if by_id[sid]["label"] else "")
                    ),
                )
                if st.button("Load into sandbox", type="primary", key=f"{ui}load_saved"):
                    _apply_to_sandbox(load(choice))

        with upload_tab:
            upload = st.file_uploader("Snapshot JSON", type=["json"], key=f"{ui}upload")
            if upload is not None and st.button("Load uploaded JSON", type="primary",
                                                key=f"{ui}load_upload"):
                try:
                    uploaded = validate(json.loads(upload.getvalue()))
                    uploaded.setdefault("id", upload.name)
                    _apply_to_sandbox(uploaded)
                except ValueError as exc:
                    st.error(f"Not a usable snapshot: {exc}")

        with edit_tab:
            st.caption(
                'Any JSON with an "inputs" object works. Keys are written WITHOUT '
                'the "bmo_" prefix, e.g. "target_production_mt": 2400. Tables use '
                "the same layout a downloaded snapshot does."
            )
            current = st.session_state.get(f"{ui}loaded_inputs") or {}
            text = st.text_area(
                "Inputs JSON", value=json.dumps({"inputs": current}, indent=1),
                height=320, key=f"{ui}json_editor",
            )
            if st.button("Apply edited JSON", type="primary", key=f"{ui}load_edit"):
                try:
                    edited = validate(json.loads(text))
                    edited.setdefault("id", "edited JSON")
                    edited["source"] = "the JSON editor"
                    _apply_to_sandbox(edited)
                except (ValueError, json.JSONDecodeError) as exc:
                    st.error(f"Not a usable snapshot: {exc}")

        if st.button("↺ Reset sandbox", key=f"{ui}reset",
                     help="Clear every sandbox input and result."):
            _reset_sandbox()

        if loaded and loaded.get("not_restorable"):
            st.caption(
                "Recorded but not restorable (Streamlit does not allow buttons or "
                "table editors to be set): " + ", ".join(loaded["not_restorable"])
            )


__all__ = ["render_snapshot_panel", "render_sandbox_loader", "SOURCE_BY_PREFIX"]
