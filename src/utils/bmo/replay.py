"""Freeze the plant data an optimiser run used, and replay it in the sandbox.

WHY INPUTS ALONE DO NOT REPRODUCE A RUN.

The LP and DE results depend on far more than what the operator types. Every
run also reads plant data the page fetches for itself:

    ore stock and chemistry        provider.build_ore_inputs        Neon, per session
    HM / slag snapshot, DPR slag   provider.get_hm_slag_snapshot     Neon, hourly
    fuel analysis (moisture, VM)   provider.get_fuel_analysis_...    Neon
    process history and context    provider.get_history_frame /      static CSV (refreshed
      (coke-rate and Si models)      get_process_context               hourly) + live Influx
    last-shift manual blend        provider.get_recent_manual_...    Neon
    live coke / nut coke / PCI     _recent_fuel_rates_*              Influx 1 h average
    configuration                  setting_bmo.yml                   file

Replaying a snapshot against today's data gives today's answer, not the saved
one: the coke rate is back-solved from a model whose features are the latest
history rows, so it moves every time the static dataset refreshes or an Influx
window closes.

RECORDING (live page). The page's provider is wrapped once (``record_provider``)
and every call's result is kept by reference. The comparison section re-reads
history on EVERY rerun, so "the latest call" is not what the run used; instead,
when a rerun ends with new LP/DE results (``finalize_run``), the calls made
BEFORE the results changed are pinned as that run's context, together with the
page's own derived values and the session values the run read at its start.

REPLAYING (sandbox). The page is executed from source (``utils/bmo/sandbox.py``)
into a namespace whose assignments pass through ``page_substitutes``: when the
page defines ``_get_context_provider``, ``_get_bmo_config``,
``_recent_fuel_rates_live`` and the like, the frozen versions are stored instead.
Anything the snapshot did not record (a chemistry mode never used, say) falls
back to live data and is listed on the page, never silently mixed in.

Nothing here changes what the live page computes: the recording wrapper returns
exactly what the provider returned.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import MutableMapping
from datetime import datetime
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from utils.bmo.sandbox import UI_SUFFIX

log = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# Session values an optimiser run READS when it starts: the manual blend and its
# Si anchor the coke correction's reference. They are written by the comparison
# section AFTER each run, so the run that produced a result saw the PREVIOUS
# values. Same two keys on main and UAT.
RUN_INPUT_SUFFIXES = ("manual_quantities_mt", "manual_si")

# Page-level values computed from files or live sources, frozen with the run.
# ``slag_settings_values`` and ``de_seed_choice`` come from widgets without keys;
# they are replayed by folding them into the frozen config those widgets default
# from (see ``frozen_config``).
PAGE_FROZEN_VARS = (
    "bmo_cfg",
    "operator_preferences",
    "recent_fuel_rates",
    "model_input_defaults",
    "slag_settings_values",
    "de_seed_choice",
)

# Session keys (after the prefix) used by this module. All sit under ``ui_`` so
# they are never captured as inputs nor restored as widget values.
K_LATEST = f"{UI_SUFFIX}ctx_latest"        # call key -> latest result
K_RERUN = f"{UI_SUFFIX}ctx_rerun"          # this rerun's calls
K_RECORD = f"{UI_SUFFIX}run_record"        # the pinned context of the last run
K_FROZEN = f"{UI_SUFFIX}frozen"            # decoded frozen context being replayed
K_PENDING = f"{UI_SUFFIX}replay_pending"   # run-start session values to replay
K_FALLBACKS = f"{UI_SUFFIX}replay_fallbacks"  # calls answered live in the sandbox


def results_marker(state: Mapping[str, Any], prefix: str) -> tuple:
    """Identity of the results on the page. Changes exactly when a run stores new ones."""

    out = []
    for suffix in ("lp_result", "de_result"):
        value = state.get(f"{prefix}{suffix}")
        out.append(None if value is None else id(value))
    return tuple(out)


def _norm_arg(value: Any) -> str:
    if isinstance(value, pd.DataFrame):
        return "<frame>"
    if isinstance(value, (list, tuple)) and value and all(hasattr(v, "ore_id") for v in value):
        return "ores:" + ",".join(sorted(str(v.ore_id) for v in value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return repr(value)
    return f"<{type(value).__name__}>"


def call_key(name: str, args: tuple, kwargs: Mapping[str, Any]) -> str:
    """A stable text key for one provider call, e.g. ``build_ore_inputs(mode='latest', window_days=30)``."""

    parts = [_norm_arg(a) for a in args]
    parts += [f"{k}={_norm_arg(v)}" for k, v in sorted(kwargs.items())]
    return f"{name}({', '.join(parts)})"


# --- recording -------------------------------------------------------------------


class RecordingProvider:
    """Pass-through wrapper that remembers what every public provider call returned."""

    def __init__(self, inner: Any, state: MutableMapping[str, Any], prefix: str) -> None:
        self._inner = inner
        self._state = state
        self._prefix = prefix
        latest = state.get(f"{prefix}{K_LATEST}")
        if not isinstance(latest, dict):
            latest = {}
            state[f"{prefix}{K_LATEST}"] = latest
        self._latest = latest
        self._rerun = {
            "start_marker": results_marker(state, prefix),
            "start_session": {s: state.get(f"{prefix}{s}") for s in RUN_INPUT_SUFFIXES},
            "calls": [],
            "finalized": False,
        }
        state[f"{prefix}{K_RERUN}"] = self._rerun

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name.startswith("_") or not callable(attr):
            return attr

        def recorded(*args: Any, **kwargs: Any) -> Any:
            result = attr(*args, **kwargs)
            key = call_key(name, args, kwargs)
            self._latest[key] = result
            self._rerun["calls"].append({
                "key": key,
                "result": result,
                "before_run": results_marker(self._state, self._prefix)
                == self._rerun["start_marker"],
            })
            return result

        return recorded


def record_provider(provider: Any, prefix: str = "bmo_") -> Any:
    """Wrap the page's provider so the data each run used can be frozen.

    Called once per rerun where the page creates its provider. Outside a
    Streamlit session (tests, scripts) the provider is returned unchanged.
    """

    try:
        import streamlit as st

        state = st.session_state
        state.get(f"{prefix}{K_LATEST}")  # raises without a running session
    except Exception:  # noqa: BLE001
        return provider
    return RecordingProvider(provider, state, prefix)


def _now_iso() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


def finalize_run(
    state: MutableMapping[str, Any], prefix: str, page_vars: Mapping[str, Any]
) -> bool:
    """At the end of a rerun: if it produced new results, pin the context it used.

    Returns:
         - return bool - True when a run was pinned in this rerun.
    """

    rerun = state.get(f"{prefix}{K_RERUN}")
    if not isinstance(rerun, dict) or rerun.get("finalized"):
        return False
    rerun["finalized"] = True
    marker = results_marker(state, prefix)
    if marker == rerun["start_marker"] or not any(m is not None for m in marker):
        return False

    calls = dict(state.get(f"{prefix}{K_LATEST}") or {})
    # The comparison section re-reads history AFTER the run in the same rerun;
    # the calls made before the results changed are the ones the run used.
    for call in rerun["calls"]:
        if call["before_run"]:
            calls[call["key"]] = call["result"]

    state[f"{prefix}{K_RECORD}"] = build_record(
        state, prefix, page_vars,
        provider_calls=calls,
        session_at_run=dict(rerun["start_session"]),
        marker=marker,
        at=_now_iso(),
    )
    # A replayed run has happened: from here on the sandbox behaves like the
    # live page, reading whatever the comparison section last wrote.
    state.pop(f"{prefix}{K_PENDING}", None)
    return True


def build_record(
    state: Mapping[str, Any],
    prefix: str,
    page_vars: Mapping[str, Any],
    *,
    provider_calls: Mapping[str, Any] | None = None,
    session_at_run: Mapping[str, Any] | None = None,
    marker: tuple | None = None,
    at: str | None = None,
) -> dict[str, Any]:
    """Everything a run depended on, held by reference (encoding happens on save)."""

    from utils.bmo.snapshot import PAGE_INPUT_TABLES, classify_suffix

    inputs: dict[str, Any] = {}
    for key in list(state.keys()):
        key = str(key)
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if classify_suffix(suffix) == "inputs":
            inputs[suffix] = state[key]
    for var, suffix in PAGE_INPUT_TABLES.items():
        table = page_vars.get(var)
        if isinstance(table, pd.DataFrame) and not table.empty:
            inputs[suffix] = table.copy()

    page = {}
    for name in PAGE_FROZEN_VARS:
        if name in page_vars:
            try:
                page[name] = copy.deepcopy(page_vars[name])
            except Exception:  # noqa: BLE001 - keep the reference if it will not copy
                page[name] = page_vars[name]

    if provider_calls is None:
        provider_calls = dict(state.get(f"{prefix}{K_LATEST}") or {})
    if session_at_run is None:
        session_at_run = {s: state.get(f"{prefix}{s}") for s in RUN_INPUT_SUFFIXES}
    return {
        "at": at or _now_iso(),
        "marker": marker if marker is not None else results_marker(state, prefix),
        "inputs": inputs,
        "page": page,
        "provider_calls": dict(provider_calls),
        "session_at_run": dict(session_at_run),
    }


# --- replay ----------------------------------------------------------------------


class ReplayProvider:
    """Answers provider calls from a snapshot; falls back to live data, and says so."""

    def __init__(self, inner: Any, calls: Mapping[str, Any], fallbacks: list[str]) -> None:
        self._inner = inner
        self._calls = calls
        self._fallbacks = fallbacks

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name.startswith("_") or not callable(attr):
            return attr

        def replayed(*args: Any, **kwargs: Any) -> Any:
            key = call_key(name, args, kwargs)
            if key in self._calls:
                return copy.deepcopy(self._calls[key])
            if key not in self._fallbacks:
                self._fallbacks.append(key)
            return attr(*args, **kwargs)

        return replayed


def frozen_config(page: Mapping[str, Any]) -> dict[str, Any] | None:
    """The run's config, with the values of the page's key-less widgets folded in.

    The Advanced Slag Balance inputs and the DE start point have no widget keys,
    so session state cannot restore them. Both take their DEFAULT from this
    config, so writing the run's values into it makes the sandbox open with them
    - and they stay editable there.
    """

    cfg = page.get("bmo_cfg")
    if not isinstance(cfg, Mapping):
        return None
    cfg = copy.deepcopy(dict(cfg))
    slag = page.get("slag_settings_values")
    if isinstance(slag, Mapping):
        section = dict(cfg.get("slag_balance") or {})
        section.update(slag)
        cfg["slag_balance"] = section
    seed = page.get("de_seed_choice")
    if seed:
        runtime_optimizer = (cfg.get("optimization_runtime") or {}).get("optimizer")
        if runtime_optimizer:
            runtime = dict(cfg["optimization_runtime"])
            runtime["optimizer"] = {**runtime_optimizer, "initial_solution": seed}
            cfg["optimization_runtime"] = runtime
        else:
            cfg["optimization"] = {**(cfg.get("optimization") or {}), "initial_solution": seed}
    return cfg


class _SandboxRefusal(RuntimeError):
    pass


def _refuse_save(*_args: Any, **_kwargs: Any) -> Any:
    raise _SandboxRefusal(
        "saving is switched off in Sandbox - it would change the live page's settings"
    )


def page_substitutes(
    frozen: Mapping[str, Any] | None, fallbacks: list[str]
) -> dict[str, Callable[[Any], Any]]:
    """Name -> function(original) -> replacement, applied as the sandbox page defines names.

    Always: preference SAVES are refused (they write the file the live page reads).
    With a frozen context: data sources are answered from it.
    """

    subs: dict[str, Callable[[Any], Any]] = {}
    for name in (
        "save_model_input_preferences", "save_ore_editor_preferences",
        "save_flux_preferences", "save_fuel_ash_preferences",
        "save_dust_preferences", "save_hm_chemistry_preferences",
    ):
        subs[name] = lambda _orig: _refuse_save

    # The dataset refresh would download and overwrite the shared furnace CSV,
    # and its status bar offers the same. Neither belongs in a replay.
    subs["_refresh_static_dataset_if_needed"] = lambda _orig: (
        lambda *a, **k: {"refreshed": False, "usable": True, "status": {}}
    )
    subs["_render_static_dataset_bar"] = lambda _orig: (lambda *a, **k: None)

    if not frozen:
        return subs
    page = frozen.get("page") or {}
    calls = frozen.get("provider_calls") or {}

    if calls:
        subs["_get_context_provider"] = lambda orig: (
            lambda: ReplayProvider(orig(), calls, fallbacks)
        )
    cfg = frozen_config(page)
    if cfg is not None:
        subs["_get_bmo_config"] = lambda _orig: (lambda: copy.deepcopy(cfg))
    if "recent_fuel_rates" in page:
        rates = dict(page["recent_fuel_rates"] or {})
        subs["_recent_fuel_rates_from_static_csv"] = lambda _orig: (lambda *a, **k: dict(rates))
        subs["_recent_fuel_rates_live"] = lambda _orig: (lambda *a, **k: {})
    if "operator_preferences" in page:
        prefs = page["operator_preferences"]
        subs["_cached_operator_preferences"] = lambda _orig: (
            lambda *a, **k: copy.deepcopy(prefs)
        )
    if "model_input_defaults" in page:
        defaults = dict(page["model_input_defaults"] or {})
        subs["_model_input_defaults_from_static_csv"] = lambda _orig: (
            lambda *a, **k: dict(defaults)
        )
    return subs


class PageNamespace(MutableMapping):
    """Module-level namespace for the sandbox page.

    ``exec`` stores the page's top-level names through ``__setitem__`` when this
    is passed as ``locals``; each write lands in the real globals dict (so the
    page's functions see it), after any substitution for that name.
    """

    def __init__(self, globals_: dict[str, Any], substitutes: Mapping[str, Callable]) -> None:
        self._g = globals_
        self._subs = dict(substitutes)

    def __getitem__(self, key: str) -> Any:
        return self._g[key]

    def __setitem__(self, key: str, value: Any) -> None:
        sub = self._subs.get(key)
        if sub is None and key.startswith("save_") and key.endswith("_preferences"):
            sub = lambda _orig: _refuse_save  # noqa: E731 - any future preference save
        if sub is not None:
            try:
                value = sub(value)
            except Exception:  # noqa: BLE001 - a broken substitute must not break the page
                log.exception("Sandbox substitute for %s failed", key)
        self._g[key] = value

    def __delitem__(self, key: str) -> None:
        del self._g[key]

    def __iter__(self):
        return iter(self._g)

    def __len__(self) -> int:
        return len(self._g)
