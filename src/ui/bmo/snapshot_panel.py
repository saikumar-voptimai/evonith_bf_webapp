"""Live / Sandbox switch, sandbox loader, and the snapshot panel of the Blend Mix Optimiser.

``render_mode_gate`` sits at the top of the optimiser page. In Live mode it only
draws the switch. In Sandbox mode it draws the loader, then runs the optimiser
page a second time from source, in its own ``testbmo_*`` state and with the
loaded snapshot's frozen plant data (``utils/bmo/sandbox.py``,
``utils/bmo/replay.py``), and stops - the live page below it never runs.

``render_snapshot_panel`` sits at the foot of the page. Both calls pass
``prefix="bmo_"``; inside the sandbox copy that string arrives as ``"testbmo_"``,
which is how each knows which page it is on.

Every widget here uses a ``<prefix>ui_`` key. Those are excluded from capture and
from restore: they are this panel's controls, not optimiser state.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from utils.bmo.replay import (
    K_FALLBACKS,
    K_FROZEN,
    K_LATEST,
    K_PENDING,
    K_RECORD,
    finalize_run,
    page_substitutes,
    results_marker,
)
from utils.bmo.sandbox import (
    LIVE_PREFIX,
    SANDBOX_PREFIX,
    UI_SUFFIX,
    is_writable_suffix,
    run_sandbox_page,
)
from utils.bmo.snapshot import (
    capture,

    frozen_state,
    restorable_state,
    results_state,
    validate,
)
from utils.bmo.snapshot_report import build_docx
from utils.bmo.snapshot_store import delete, list_snapshots, load, save

log = logging.getLogger(__name__)

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MODE_KEY = f"{LIVE_PREFIX}{UI_SUFFIX}sandbox"
SB_UI = f"{SANDBOX_PREFIX}{UI_SUFFIX}"
LOADED_KEY = f"{SB_UI}loaded"
LOAD_ERROR_KEY = f"{SB_UI}load_error"


# --- helpers -----------------------------------------------------------------------


def _mtime(path: str) -> int:
    try:
        return Path(path).stat().st_mtime_ns
    except OSError:
        return 0


@st.cache_data(show_spinner=False, max_entries=16)
def _docx_bytes(snapshot_id: str, path: str, mtime_ns: int) -> bytes:
    """The report for one snapshot. Cached on the file's mtime - rebuilt only if
    the snapshot changes, not on every rerun of a long page."""

    return build_docx(load(snapshot_id))


def _when(iso: str | None) -> str:
    return (iso or "").replace("T", " ")[:19]


def _label(row: Mapping[str, Any]) -> str:
    return (
        f"{_when(row['created_at'])} · {row['source']}"
        + (f" · {row['label']}" if row["label"] else "")
    )


def _table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    def blend(shares: Mapping[str, float]) -> str:
        return ", ".join(f"{k} {v:.1f}%" for k, v in list(shares.items())[:5])

    return pd.DataFrame([
        {
            "Taken at": _when(r["created_at"]),
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
            "Plant data": "frozen" if r["summary"].get("frozen_context") else "not saved",
        }
        for r in rows
    ])


_TABLE_CONFIG = {
    "Fuel rate": st.column_config.NumberColumn("Fuel rate (kg/THM)", format="%.2f"),
    "Coke rate": st.column_config.NumberColumn("Coke rate (kg/THM)", format="%.2f"),
    "Slag rate": st.column_config.NumberColumn("Slag rate (kg/THM)", format="%.1f"),
    "B2": st.column_config.NumberColumn("B2", format="%.3f"),
    "T-basicity": st.column_config.NumberColumn("T-basicity", format="%.3f"),
    "Production": st.column_config.NumberColumn("Production (MT)", format="%.0f"),
    "Total cost": st.column_config.NumberColumn("Total cost (Rs/THM)", format="%.0f"),
}


# --- loading a snapshot into the sandbox -------------------------------------------


def _clear_sandbox(state: Any) -> None:
    """Remove every sandbox input, result, cache and replay record.

    The loader's own widgets are kept so the selection survives the load.
    """

    keep = (
        f"{SB_UI}saved_choice", f"{SB_UI}upload", f"{SB_UI}label", f"{SB_UI}choice",
    )
    for key in [str(k) for k in list(state.keys())]:
        if key in keep:
            continue
        if key.startswith(SANDBOX_PREFIX) or key.startswith(f"_{SANDBOX_PREFIX}"):
            del state[key]


def apply_to_sandbox(snapshot: Mapping[str, Any]) -> None:
    """Replace the sandbox state with this snapshot: inputs, results, frozen plant data.

    Safe to call from a widget callback (no rerun inside). With a frozen context
    the sandbox also re-applies the session values the run read at its start
    until the first run there - so running LP/DE right after loading reproduces
    the saved numbers, not ones influenced by what the page displayed since.
    """

    state = st.session_state
    _clear_sandbox(state)

    inputs = restorable_state(snapshot, prefix=SANDBOX_PREFIX)
    results = results_state(snapshot, prefix=SANDBOX_PREFIX)
    frozen = frozen_state(snapshot)
    for key, value in {**inputs, **results}.items():
        state[key] = value

    state[f"{SANDBOX_PREFIX}{K_FALLBACKS}"] = []
    if frozen:
        state[f"{SANDBOX_PREFIX}{K_FROZEN}"] = frozen
        state[f"{SANDBOX_PREFIX}{K_LATEST}"] = dict(frozen["provider_calls"])
        if frozen.get("session_at_run") and frozen["provider_calls"]:
            state[f"{SANDBOX_PREFIX}{K_PENDING}"] = dict(frozen["session_at_run"])
        # A snapshot taken straight after loading reproduces the loaded one.
        state[f"{SANDBOX_PREFIX}{K_RECORD}"] = {
            "at": (snapshot.get("run") or {}).get("at") or snapshot.get("created_at"),
            "marker": results_marker(state, SANDBOX_PREFIX),
            "inputs": {k[len(SANDBOX_PREFIX):]: v for k, v in inputs.items()},
            "page": dict(frozen["page"]),
            "provider_calls": dict(frozen["provider_calls"]),
            "session_at_run": dict(frozen.get("session_at_run") or {}),
        }

    state[LOADED_KEY] = {
        "id": snapshot.get("id") or "(unsaved JSON)",
        "created_at": snapshot.get("created_at", ""),
        "run_at": (snapshot.get("run") or {}).get("at") or "",
        "source": snapshot.get("source", ""),
        "label": snapshot.get("label", ""),
        "summary": snapshot.get("summary") or {},
        "restored_inputs": len(inputs),
        "restored_results": len(results),
        "frozen": bool(frozen and frozen.get("provider_calls")),
        "model_check": (snapshot.get("frozen") or {}).get("model_check"),
        "not_restorable": list(snapshot.get("not_restorable") or []),
    }
    state[f"{SB_UI}json_text"] = json.dumps({"inputs": snapshot.get("inputs") or {}}, indent=1)
    state.pop(LOAD_ERROR_KEY, None)


def _load_saved_cb(choice_key: str) -> None:
    snapshot_id = st.session_state.get(choice_key)
    if not snapshot_id:
        return
    try:
        apply_to_sandbox(load(snapshot_id))
        st.session_state[MODE_KEY] = True
    except Exception as exc:  # noqa: BLE001
        st.session_state[LOAD_ERROR_KEY] = f"Could not load {snapshot_id}: {exc}"


def _load_upload_cb() -> None:
    upload = st.session_state.get(f"{SB_UI}upload")
    if upload is None:
        return
    try:
        snapshot = validate(json.loads(upload.getvalue()))
        snapshot.setdefault("id", upload.name)
        apply_to_sandbox(snapshot)
    except (ValueError, json.JSONDecodeError) as exc:
        st.session_state[LOAD_ERROR_KEY] = f"Not a usable snapshot: {exc}"


def _load_edited_cb() -> None:
    text = st.session_state.get(f"{SB_UI}json_text") or ""
    try:
        snapshot = validate(json.loads(text))
        snapshot.setdefault("id", "edited JSON")
        snapshot["source"] = "the JSON editor"
        apply_to_sandbox(snapshot)
    except (ValueError, json.JSONDecodeError) as exc:
        st.session_state[LOAD_ERROR_KEY] = f"Not a usable snapshot: {exc}"


def _reset_cb() -> None:
    _clear_sandbox(st.session_state)
    st.session_state.pop(LOADED_KEY, None)


# --- the gate -------------------------------------------------------------------------


def _preserve(state: Any, prefix: str) -> None:
    """Keep the other mode's inputs while its widgets are not drawn.

    Streamlit drops the state of any widget not rendered in a run. Writing a
    value back through the session-state API stores it under its key instead,
    where it survives until the widget is drawn again. Buttons, editors and
    uploaders refuse that write, so they are left alone.
    """

    for key in [str(k) for k in list(state.keys())]:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if suffix.startswith(UI_SUFFIX) or not is_writable_suffix(suffix):
            continue
        try:
            state[key] = state[key]
        except Exception:  # noqa: BLE001 - never block the page over a convenience
            pass


def render_mode_gate(prefix: str = "bmo_") -> None:
    """Top of the optimiser page: Live / Sandbox switch; in Sandbox, run the replay and stop."""

    if prefix != LIVE_PREFIX:
        return  # this is the sandbox copy of the page running
    state = st.session_state
    sandbox = st.toggle(
        "🧪 Sandbox",
        key=MODE_KEY,
        help=(
            "Sandbox replays a saved snapshot - inputs, results and the plant data "
            "the run used - in a separate copy of this page. Change anything and run "
            "it; the live page is not touched."
        ),
    )
    _preserve(state, LIVE_PREFIX if sandbox else SANDBOX_PREFIX)
    if not sandbox:
        return

    _render_loader(state)
    pending = state.get(f"{SANDBOX_PREFIX}{K_PENDING}")
    if isinstance(pending, dict):
        for suffix, value in pending.items():
            state[f"{SANDBOX_PREFIX}{suffix}"] = value
    fallbacks = state.setdefault(f"{SANDBOX_PREFIX}{K_FALLBACKS}", [])
    frozen = state.get(f"{SANDBOX_PREFIX}{K_FROZEN}")
    footer = st.empty()
    try:
        run_sandbox_page(page_substitutes(frozen, fallbacks))
    finally:
        if frozen and fallbacks:
            footer.caption(
                "Answered from LIVE data because the snapshot did not record them: "
                + "; ".join(fallbacks)
            )
    st.stop()


def _render_loader(state: Any) -> None:
    loaded = state.get(LOADED_KEY)
    if not loaded:
        st.info(
            "🧪 **Sandbox** — a separate copy of this page. Load a snapshot to replay "
            "it exactly as saved, or start from defaults. Nothing here changes the "
            "live page or its saved settings."
        )
    else:
        s = loaded["summary"] or {}
        lines = [
            f"🧪 **Sandbox — {loaded['id']}**"
            + (f" · *{loaded['label']}*" if loaded["label"] else "")
            + (f" · run at {_when(loaded['run_at'])}" if loaded["run_at"] else "")
            + (f" · taken on {loaded['source']}" if loaded["source"] else ""),
        ]
        if loaded["frozen"]:
            lines.append(
                "Plant data is **frozen as saved** — ore stock and chemistry, HM/slag, "
                "fuel analysis, process history for the coke-rate and Si models, live "
                "fuel rates and configuration. Run LP/DE to reproduce the saved result."
            )
        else:
            lines.append(
                "⚠️ This snapshot carries **no frozen plant data** (taken before "
                "freezing existed, or a hand-written JSON). Its inputs are applied to "
                "TODAY's plant data, so results will differ from the saved ones."
            )
        if s.get("result"):
            lines.append(
                f"Saved result ({s['result']}): total cost "
                f"{(s.get('total_cost_rs_thm') or 0):,.0f} Rs/THM · coke "
                f"{(s.get('coke_rate_kg_thm') or 0):,.2f} kg/THM · fuel "
                f"{(s.get('fuel_rate_kg_thm') or 0):,.2f} kg/THM · slag "
                f"{(s.get('slag_rate_kg_thm') or 0):,.1f} kg/THM."
            )
        (st.success if loaded["frozen"] else st.warning)("  \n".join(lines))
        if loaded.get("not_restorable"):
            st.caption(
                "Recorded but not restorable (Streamlit does not let buttons or table "
                "editors be set): " + ", ".join(loaded["not_restorable"])
            )

    if state.get(LOAD_ERROR_KEY):
        st.error(state[LOAD_ERROR_KEY])

    rows, _broken = list_snapshots()
    by_id = {r["id"]: r for r in rows}
    pick_col, load_col, reset_col = st.columns([4, 1, 1], vertical_alignment="bottom")
    if by_id:
        pick_col.selectbox(
            "Snapshot to replay", list(by_id), key=f"{SB_UI}saved_choice",
            format_func=lambda sid: _label(by_id[sid]),
        )
        load_col.button(
            "Load", type="primary", width="stretch", key=f"{SB_UI}load_saved",
            on_click=_load_saved_cb, args=(f"{SB_UI}saved_choice",),
        )
    else:
        pick_col.caption("No snapshots saved yet — take one in Live mode.")
    reset_col.button(
        "↺ Reset", width="stretch", key=f"{SB_UI}reset", on_click=_reset_cb,
        help="Clear every sandbox input, result and frozen data.",
    )

    with st.expander("Upload or edit a snapshot JSON", expanded=False):
        upload_tab, edit_tab = st.tabs(["Upload JSON", "Edit JSON"])
        with upload_tab:
            st.file_uploader("Snapshot JSON", type=["json"], key=f"{SB_UI}upload")
            st.button("Load uploaded JSON", key=f"{SB_UI}load_upload", on_click=_load_upload_cb)
        with edit_tab:
            st.caption(
                'Any JSON with an "inputs" object works. Keys are written WITHOUT the '
                '"bmo_" prefix, e.g. "target_production_mt": 2400. A hand-written JSON '
                "has no frozen plant data, so it runs on today's data."
            )
            state.setdefault(f"{SB_UI}json_text", json.dumps({"inputs": {}}, indent=1))
            st.text_area("Inputs JSON", height=260, key=f"{SB_UI}json_text")
            st.button("Apply edited JSON", key=f"{SB_UI}load_edit", on_click=_load_edited_cb)
    st.divider()


# --- the snapshot panel -------------------------------------------------------------------


def _open_in_sandbox_cb(snapshot_id: str) -> None:
    try:
        apply_to_sandbox(load(snapshot_id))
        st.session_state[MODE_KEY] = True
        st.session_state[f"{SB_UI}saved_choice"] = snapshot_id
    except Exception as exc:  # noqa: BLE001
        st.session_state[LOAD_ERROR_KEY] = f"Could not load {snapshot_id}: {exc}"


def render_snapshot_panel(
    prefix: str = "bmo_", page_vars: Mapping[str, Any] | None = None
) -> None:
    """Take Snapshot, the snapshot table, and the report download.

    Also closes the rerun for the replay module: a run finished in this rerun
    has its plant data pinned here, before anything else can change it.

    Args:
         - prefix: str - State namespace of the page calling this.
         - page_vars: Mapping | None - The calling page's globals, so the snapshot
           records the input tables the page ACTUALLY ran with.
    """

    page_vars = page_vars or {}
    state = st.session_state
    try:
        finalize_run(state, prefix, page_vars)
    except Exception:  # noqa: BLE001 - recording must never take the page down
        log.exception("Could not pin the run context")

    ui = f"{prefix}{UI_SUFFIX}"
    in_sandbox = prefix == SANDBOX_PREFIX
    st.divider()
    st.markdown("### 📸 Snapshots")
    st.caption(
        "A snapshot records the last run: every input, the LP/DE results, and the "
        "plant data the run read (chemistry, stock, HM/slag, process history for the "
        "coke-rate and Si models, live fuel rates, configuration). Download it as a "
        "report, or open it in Sandbox to replay it exactly."
    )

    label_col, button_col = st.columns([3, 1], vertical_alignment="bottom")
    label = label_col.text_input(
        "Label (optional)", key=f"{ui}label",
        placeholder="e.g. Shift B — trial with 12% pellet",
    )
    if button_col.button("📸 Take Snapshot", type="primary", width="stretch",
                         key=f"{ui}take"):
        try:
            snapshot = capture(state, prefix=prefix, page_vars=page_vars, label=label)
            path = save(snapshot)
            st.success(f"Snapshot **{snapshot['id']}** saved — {path.name}")
            run = snapshot.get("run") or {}
            if not snapshot["results"]:
                st.info("No optimiser result was on the page, so this snapshot holds "
                        "inputs only. Run LP or DE first for a full report.")
            elif run.get("pinned"):
                st.caption(f"Recorded the run of {_when(run.get('at'))} with its plant data.")
            if run.get("inputs_changed_after_run"):
                st.warning(
                    "Inputs were changed after the last run. The snapshot holds the "
                    "inputs THAT RUN used, so it matches its results — re-run first to "
                    "record the new inputs."
                )
            check = (snapshot.get("frozen") or {}).get("model_check")
            if check and not check.get("identical"):
                st.warning(
                    "The saved process history does not reproduce the fuel model "
                    f"exactly ({check['model_output_saved_history']:.4f} vs "
                    f"{check['model_output_full_history']:.4f} Rs/THM)."
                )
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
        format_func=lambda sid: _label(by_id[sid]),
    )
    row = by_id[choice]
    cols = st.columns(4 if not in_sandbox else 3)
    try:
        report = _docx_bytes(choice, row["path"], _mtime(row["path"]))
        cols[0].download_button(
            "⬇️ Download Snapshot (.docx)", data=report,
            file_name=f"BMO_snapshot_{choice}.docx", mime=DOCX_MIME,
            width="stretch", key=f"{ui}download_docx",
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Report build failed")
        cols[0].error(f"Report could not be built: {exc}")
    try:
        with open(row["path"], "rb") as handle:
            cols[1].download_button(
                "⬇️ Download JSON", data=handle.read(),
                file_name=f"BMO_snapshot_{choice}.json", mime="application/json",
                width="stretch", key=f"{ui}download_json",
            )
    except OSError as exc:
        cols[1].error(str(exc))
    if not in_sandbox:
        cols[2].button(
            "🧪 Open in Sandbox", width="stretch", key=f"{ui}open_sandbox",
            on_click=_open_in_sandbox_cb, args=(choice,),
        )
    if cols[-1].button("🗑️ Delete", width="stretch", key=f"{ui}delete"):
        delete(choice)
        st.rerun()


__all__ = ["render_mode_gate", "render_snapshot_panel", "apply_to_sandbox"]
