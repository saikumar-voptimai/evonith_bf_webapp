"""Golden runs of the Blend Mix Optimiser page (page 9) on frozen plant data (SV-3.4, #301).

WHAT IS RECORDED. For each golden day the page source runs, unchanged, under Streamlit's ``AppTest``: first load,
"Run Balanced Optimizer", then "Take Snapshot". The record holds what page 9 produces for that day:

    plan       the LP plan (``bmo_lp_result``), the manual blend it is compared with, the coke anchor
    tables     every element of the result section (Balanced / Intensive / Comparison tabs, the HM Si panel, the
               coke-rate prediction and the charged-coke panel): tables, metrics, captions, alerts, chart traces
    charged    the charged-coke panel payload (prediction, issued record, the 7-day replay)
    si         the HM Si record and the optimiser tile built from it
    snapshot   the snapshot the page saves (inputs, results, context, summary; the frozen plant data by digest)

HOW THE DAY IS FROZEN. The page is executed the way its Sandbox replays a snapshot (``utils/bmo/replay.py``): its
top-level names pass through ``PageNamespace`` substitutions. Every source of the day is a committed fixture:

    yard, chemistry, HM/slag, fuel analysis, process history,    frozen provider calls of a committed snapshot
    last-shift blend, operator inputs and preferences            (``fixtures/bmo_golden/plant_*.json.gz``)
    configuration                                                ``fixtures/bmo_golden/setting_bmo.json``
    hourly dataset (charged-coke forecast, last-shift baseline)  ``fixtures/charged_coke/source_window.csv.gz``,
                                                                 cut at the day's last complete row
    HM Si sources                                                ``fixtures/bmo_si_forecast``
    clock                                                        the day's wall-clock time (``pd.Timestamp.now``)
    live process tags (energy balance, control recommendation)   the day's last hourly row (``live_tags`` cases),
                                                                 else unavailable, as when InfluxDB is down

InfluxDB, Neon and the network are never reached: the live fetchers raise, ledgers and snapshots are written to a
temporary folder. Model accuracy, commentary and SBFE are not part of the record and are stubbed (they read the
local storage folders).

Rebuild after an intended change of page-9 output (review the diff of the JSON files):

    python -m tests.bmo_golden --write          (from the checkout root, with src/ importable)
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import functools
import gzip
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import types
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import pandas

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
PAGE = SRC / "custom_pages" / "9_Blend_Optimizer.py"
FIXTURES = REPO / "tests" / "fixtures" / "bmo_golden"
CHARGED_SOURCE = REPO / "tests" / "fixtures" / "charged_coke" / "source_window.csv.gz"
SI_FIXTURE = REPO / "tests" / "fixtures" / "bmo_si_forecast"
PLANT_TZ = "Asia/Kolkata"
PREFIX = "bmo_"
#: The environment of tests/conftest.py that page 9 reads (plan day of the blast moisture, no Open-Meteo request).
TEST_ENVIRONMENT = {
    "COKE_ENERGY_WEATHER_LIVE": "0",
    "COKE_ENERGY_TODAY": "2026-10-15",
    "BF_TWIN_DB_URL": "off",
    "QDRANT_URL": "http://localhost:6333",
    "OPENROUTER_API_KEY": "test-openrouter-key",
}
#: Relative tolerance of the comparison (the programme's parity rule, SV-x.y: <= 1e-6).
RTOL = 1e-6
ATOL = 1e-9


@dataclass(frozen=True)
class Case:
    """One golden day: the plant state of a committed snapshot, run at the day's clock."""

    day: str  # golden file name, the plant day (IST)
    plant: str  # fixtures/bmo_golden/plant_<plant>.json.gz
    now: str  # IST wall clock of the run (the charged-coke row is the hour before; Si origin = now floored)
    coke_model: str = "Data-Driven"  # or "Physics-Driven"
    inputs: Mapping[str, Any] = field(default_factory=dict)  # session values (without prefix) set on top
    live_tags: bool = False  # the live process tags: the day's last hourly row (else unavailable, as offline)


#: The days lie after the shipped charged-coke model's training cutoff (30 Sep 2026 04:00), so its 7-day replay has
#: rows, and inside the hourly (to 4 Oct) and Si (to 6 Oct) fixtures; at each clock the charged-coke forecast is shown
#: (Data-Driven runs are not refused). The yard is a committed snapshot's (24 and 25 Sep 2026).
CASES: tuple[Case, ...] = (
    Case("2026-09-30", "20260924_175523", "2026-09-30 22:10"),
    Case("2026-10-01", "20260925_135718", "2026-10-01 13:10"),
    Case("2026-10-02", "20260925_225150", "2026-10-02 14:10"),
    Case("2026-10-03", "20260925_225537", "2026-10-03 20:10", live_tags=True),
    Case(
        "2026-10-04",
        "20260925_225537",
        "2026-10-04 09:10",
        coke_model="Physics-Driven",
        inputs={"target_production_mt": 2300.0},
        live_tags=True,
    ),
)

#: Live process tag (``_live_process_snapshot``) <- hourly dataset column, for the cases with ``live_tags``.
LIVE_TAG_COLUMNS = {
    "hot_blast_vol_nm3h": "HOT BLAST VOLUMENM3/HR.",
    "hot_blast_temp": "HOT BLAST TEMP.OC",
    "hot_blast_press": "HOT BLAST PRESSUREBAR",
    "oxygen_enrichment_pct": "O2 ENRICHMENT %",
    "top_press_avg": "TOPPRESSUREBAR",
    "steam_injection": "STEAMKGS/HR.",
    "co_pct": "FURNACE TOP GAS ANALYSISONLINE (ANALYZER)CO%",
    "co2_pct": "FURNACE TOP GAS ANALYSISCO2%",
    "h2_pct": "FURNACE TOP GAS ANALYSISH2%",
    "top_temp_avg": "FTG_UPTAKE_TEMP_AVG",
    "coal_rate_actual_value": "PCI_KG/THM",
    "oxygen_flow": "OXYGENFLOWNM3/HR.",
    "production_per_hour": "PRODUCTIONTONNESPERHR",
    "shell_loss_gj_per_hr": "TOTAL HEAT LOAD",
}


# --------------------------------------------------------------------------------------------- frozen clock
def _frozen_now(now_utc: pandas.Timestamp):
    def now(tz: Any = None) -> pandas.Timestamp:
        return now_utc.tz_convert(tz) if tz is not None else now_utc.tz_convert(PLANT_TZ).tz_localize(None)

    return now


class _TimestampMeta(type):
    """``Timestamp`` of the frozen-clock pandas: constructs and type-checks as pandas', only ``now`` is fixed."""

    def __call__(cls, *args: Any, **kwargs: Any) -> pandas.Timestamp:
        return pandas.Timestamp(*args, **kwargs)

    def __instancecheck__(cls, obj: Any) -> bool:
        return isinstance(obj, pandas.Timestamp)

    def __getattr__(cls, name: str) -> Any:
        return getattr(pandas.Timestamp, name)


def frozen_pandas(now_utc: pandas.Timestamp) -> types.ModuleType:
    """The pandas module with ``Timestamp.now`` fixed at ``now_utc``; everything else is pandas itself."""

    stamp = _TimestampMeta("Timestamp", (), {"now": staticmethod(_frozen_now(now_utc))})

    class _Pandas(types.ModuleType):
        def __getattr__(self, name: str) -> Any:
            return getattr(pandas, name)

    module = _Pandas("pandas")
    module.Timestamp = stamp
    return module


# --------------------------------------------------------------------------------------------- fixtures
def load_plant(name: str) -> dict[str, Any]:
    with gzip.open(FIXTURES / f"plant_{name}.json.gz", "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_config() -> dict[str, Any]:
    return json.loads((FIXTURES / "setting_bmo.json").read_text(encoding="utf-8"))


def golden_path(case: Case) -> Path:
    return FIXTURES / f"{case.day}.json.gz"


def load_golden(case: Case) -> dict[str, Any]:
    with gzip.open(golden_path(case), "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _day_dataset(case: Case, folder: Path) -> Path:
    """The hourly dataset as published at the case's clock: rows up to the hour before, build time HH:06."""

    now = pandas.Timestamp(case.now)
    built = now.floor("h") + pandas.Timedelta(minutes=6)
    frame = pandas.read_csv(CHARGED_SOURCE, float_precision="round_trip")
    stamps = pandas.to_datetime(frame["time"])
    frame = frame.loc[stamps <= built.floor("h") - pandas.Timedelta(hours=1)]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "furnace_dataset.csv"
    frame.to_csv(path, index=False)
    meta = {"csv_file": "", "source_last_modified": format_datetime(built.tz_localize(PLANT_TZ).to_pydatetime())}
    (folder / "cache_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


def _live_tags(case: Case, dataset: Path) -> dict[str, float]:
    """The page's 1-hour live process snapshot for the day: the last row of the day's hourly dataset."""

    if not case.live_tags:
        return {}
    row = pandas.read_csv(dataset, usecols=list(LIVE_TAG_COLUMNS.values()), float_precision="round_trip").iloc[-1]
    return {tag: float(row[column]) for tag, column in LIVE_TAG_COLUMNS.items() if pandas.notna(row[column])}


def _si_sources(fetched_at: pandas.Timestamp):
    from data.bmo.si_forecast_context import SiliconForecastSources

    kwargs = {"float_precision": "round_trip"}
    return SiliconForecastSources(
        online=pandas.read_csv(SI_FIXTURE / "online_10min.csv.gz", **kwargs),
        labs=pandas.read_csv(SI_FIXTURE / "raw_hm_slag.csv", **kwargs),
        charge=pandas.read_csv(SI_FIXTURE / "raw_charge.csv.gz", **kwargs),
        hourly_gate=pandas.read_csv(SI_FIXTURE / "hourly_gate_source.csv", **kwargs),
        fetched_at=pandas.Timestamp(fetched_at).tz_convert("UTC"),
    )


class _DayProvider:
    """Provider calls the frozen snapshot does not answer.

    The last-shift blend is read from the day's hourly dataset by the provider's own method (the page passes
    ``lookback_hours``, which selects that source); top-gas dust has a fixed answer (no DPR); any other call fails
    the run, so a new data read on the page shows up here instead of reaching Neon.
    """

    def __init__(self, dataset: Path) -> None:
        from data.bmo import EvonithBmoContextProvider

        self._real = EvonithBmoContextProvider(
            setting_path=str(SRC / "config" / "setting_bmo.yml"),
            mapping_path=str(SRC / "config" / "bmo_ore_mapping.yml"),
        )
        self._real._dataset_service.static_dataset_path = str(dataset)

    def get_recent_manual_blend_snapshot(self, selected_ores: Any, lookback_hours: Any = None) -> dict[str, Any]:
        if lookback_hours is None:
            raise AssertionError("the golden runs read the last-shift blend from the hourly dataset only")
        return self._real.get_recent_manual_blend_snapshot(selected_ores, lookback_hours=lookback_hours)

    def get_dpr_dust_rates(self, **_kwargs: Any) -> tuple[dict, list[str]]:
        return {}, []

    def get_charge_mix_snapshot(self) -> dict:
        return {}

    def get_data_diagnostics(self) -> dict:
        return {}

    def __getattr__(self, name: str) -> Any:
        def missing(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(f"provider call not in the golden fixture: {name}{args}{kwargs}")

        return missing


# --------------------------------------------------------------------------------------------- the run
#: Set by ``run_case`` before each AppTest run; read by the page script (same process).
CURRENT: dict[str, Any] = {}


def _script() -> None:  # pragma: no cover - executed by AppTest
    from tests.bmo_golden import exec_page

    exec_page()


def exec_page() -> None:
    """Execute page 9 with the current case's substitutions (called inside the AppTest script run)."""

    from utils.bmo.replay import PageNamespace, ReplayProvider, page_substitutes

    case: Case = CURRENT["case"]
    frozen = CURRENT["frozen"]
    fallbacks: list[str] = CURRENT["fallbacks"]
    subs = page_substitutes(frozen, fallbacks)
    # The last-shift blend of the DAY: the snapshot's own answer (another day, another source) is not replayed.
    calls = {k: v for k, v in frozen["provider_calls"].items() if not k.startswith("get_recent_manual_blend_snapshot(")}
    day_provider = _DayProvider(CURRENT["dataset"])
    subs["_get_context_provider"] = lambda _orig: lambda: ReplayProvider(day_provider, calls, fallbacks)
    dataset = str(CURRENT["dataset"])
    subs["_static_dataset_cache_token"] = lambda _orig: lambda *_a, **_k: (dataset, 1)
    live_tags = CURRENT["live_tags"]
    subs["_live_process_snapshot"] = lambda _orig: lambda: dict(live_tags)
    subs["pd"] = lambda _real: CURRENT["pandas"]
    subs["LiveSiliconForecastSource"] = lambda _orig: CURRENT["si_source"]
    # The pellet chemistry age is judged at the day's clock, not at the clock of the test run.
    now_utc = CURRENT["now_utc"]
    subs["validate_selected_pellet_inputs"] = lambda orig: functools.partial(orig, now=now_utc.to_pydatetime())
    subs["is_logged_in"] = lambda _orig: lambda: True
    subs["is_admin"] = lambda _orig: lambda: False
    subs["is_supervisor"] = lambda _orig: lambda: False
    for name in ("render_model_accuracy_tab", "render_furnace_commentary", "render_production_frontier"):
        subs[name] = lambda _orig: lambda *_a, **_k: None
    del case
    code = compile(PAGE.read_text(encoding="utf-8-sig"), str(PAGE), "exec", dont_inherit=True)
    namespace: dict[str, Any] = {"__name__": "__bmo_golden__", "__file__": str(PAGE), "__builtins__": builtins}
    CURRENT["namespace"] = namespace
    exec(code, namespace, PageNamespace(namespace, subs))  # noqa: S102 - the page source of this checkout


def _si_source_class(now_utc: pandas.Timestamp) -> type:
    class FixtureSiSource:
        """``LiveSiliconForecastSource`` on the committed Si fixtures, fetched at the case's clock."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def fetch(self, origin: Any) -> Any:
            return _si_sources(now_utc)

        def fetch_range(self, **_kwargs: Any) -> Any:
            return _si_sources(now_utc)

    return FixtureSiSource


@contextlib.contextmanager
def _patched(case: Case, tmp: Path) -> Iterator[None]:
    """Process-wide patches for one run: no InfluxDB, temporary stores, the case's clock in the forecast modules."""

    import furnace_data.influx.online as influx_online
    import services.forecasts as forecasts
    import ui.bmo.si_forecast as si_ui
    import utils.bmo.si_forecast.service as si_service
    import utils.bmo.snapshot as snapshot_module
    import utils.bmo.snapshot_store as snapshot_store
    import utils.energy_balance.assumptions as assumptions
    from data.bmo import si_forecast_context

    now_utc = pandas.Timestamp(case.now).tz_localize(PLANT_TZ).tz_convert("UTC")
    clock = frozen_pandas(now_utc)
    source_class = _si_source_class(now_utc)

    def no_influx(*_args: Any, **_kwargs: Any) -> Any:
        raise ConnectionError("InfluxDB is not reachable in the golden runs")

    patches = [
        (influx_online, "fetch_online_df", no_influx),
        (snapshot_store, "STORE_DIR", tmp / "snapshots"),
        (assumptions, "OVERRIDES_PATH", tmp / "energy_balance_assumptions.json"),
        (snapshot_module, "_git_provenance", lambda: {}),
        (forecasts, "pd", clock),
        (si_service, "pd", clock),
        (si_ui, "pd", clock),
        (si_forecast_context.LiveSiliconForecastSource, "__init__", source_class.__init__),
        (si_forecast_context.LiveSiliconForecastSource, "fetch", source_class.fetch),
        (si_forecast_context.LiveSiliconForecastSource, "fetch_range", source_class.fetch_range),
    ]
    saved = [(target, name, target.__dict__.get(name, _MISSING)) for target, name, _ in patches]
    try:
        for target, name, value in patches:
            setattr(target, name, value)
        CURRENT.update(pandas=clock, si_source=source_class)
        yield
    finally:
        for target, name, value in reversed(saved):
            if value is _MISSING:
                delattr(target, name)
            else:
                setattr(target, name, value)


_MISSING = object()


def _case_config(tmp: Path, dataset: Path) -> dict[str, Any]:
    cfg = load_config()
    cfg["forecast_issuer"] = "page"
    ddc = cfg.setdefault("data_driven_coke", {})
    ddc["dataset_path"] = str(dataset)
    ddc.setdefault("charged_4h", {})["storage_dir"] = str(tmp / "charged_coke")
    si = cfg.setdefault("si_forecast", {})
    si["storage_dir"] = str(tmp / "si_forecast")
    si["static_dataset_path"] = str(dataset)
    cfg.setdefault("data_sources", {})["static_dataset_path"] = str(dataset)
    cfg["data_sources"]["static_dataset_url"] = ""
    return cfg


def run_case(case: Case, *, timeout: float = 900.0) -> dict[str, Any]:
    """Run page 9 for one golden day and return its record (see the module docstring)."""

    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from utils.bmo.snapshot import restorable_state

    plant = load_plant(case.plant)
    with tempfile.TemporaryDirectory(prefix="bmo_golden_") as tmp_name:
        tmp = Path(tmp_name)
        dataset = _day_dataset(case, tmp / "dataset")
        cfg = _case_config(tmp, dataset)
        frozen = _decode_frozen(plant)
        frozen["page"]["bmo_cfg"] = cfg
        fallbacks: list[str] = []
        CURRENT.clear()
        now_utc = pandas.Timestamp(case.now).tz_localize(PLANT_TZ).tz_convert("UTC")
        CURRENT.update(
            case=case,
            frozen=frozen,
            fallbacks=fallbacks,
            dataset=dataset,
            now_utc=now_utc,
            live_tags=_live_tags(case, dataset),
        )
        st.cache_data.clear()
        st.cache_resource.clear()
        with _patched(case, tmp):
            at = AppTest.from_function(_script, default_timeout=timeout)
            for key, value in restorable_state(plant, prefix=PREFIX).items():
                at.session_state[key] = value
            for suffix, value in frozen.get("session_at_run", {}).items():
                at.session_state[f"{PREFIX}{suffix}"] = value
            at.session_state[f"{PREFIX}coke_rate_model"] = case.coke_model
            # The operator has reviewed the selected pellet inputs (the run is refused otherwise).
            at.session_state[f"{PREFIX}confirm_pellet_inputs"] = True
            for suffix, value in case.inputs.items():
                at.session_state[f"{PREFIX}{suffix}"] = value
            at.run()
            _raise_on_exception(at, "first load")
            next(b for b in at.button if b.label == "Run Balanced Optimizer").click()
            at.run()
            _raise_on_exception(at, "Run Balanced Optimizer")
            record = _record(at, CURRENT["namespace"])
            at.button(key=f"{PREFIX}ui_take").click()
            at.run()
            _raise_on_exception(at, "Take Snapshot")
            scrub = _scrubber(tmp)
            record = scrub(normalise(record))
            record["snapshot"] = _snapshot_record(tmp / "snapshots", scrub)
        record["provider_fallbacks"] = sorted(set(fallbacks))
        record["case"] = {
            "day": case.day,
            "plant": case.plant,
            "now": case.now,
            "coke_model": case.coke_model,
            "inputs": dict(case.inputs),
            "live_tags": case.live_tags,
        }
    return normalise(record)


def _scrubber(tmp: Path):
    """Replace the run's temporary folder and the checkout's path by placeholders (they differ between runs)."""

    pairs = []
    for path, token in ((tmp, "<tmp>"), (REPO, "<repo>")):
        for form in {str(path), str(path.resolve()), path.as_posix(), path.resolve().as_posix()}:
            pairs.append((json.dumps(form)[1:-1], token))
    pairs.sort(key=lambda pair: -len(pair[0]))
    # The rest of a scrubbed path with forward slashes, so a record written on Windows compares on Linux.
    tail = re.compile(r'(<tmp>|<repo>)((?:(?:\\\\|/)[^"\\/]+)+)')

    def scrub(value: Any) -> Any:
        text = json.dumps(value, ensure_ascii=False)
        for old, new in pairs:
            text = text.replace(old, new)
        text = tail.sub(lambda m: m.group(1) + m.group(2).replace("\\\\", "/"), text)
        return json.loads(text)

    return scrub


def _decode_frozen(plant: Mapping[str, Any]) -> dict[str, Any]:
    from utils.bmo.snapshot import decode

    frozen = plant.get("frozen") or {}
    return {
        "provider_calls": {k: decode(v) for k, v in (frozen.get("provider_calls") or {}).items()},
        "page": {k: decode(v) for k, v in (frozen.get("page") or {}).items()},
        "session_at_run": {k: decode(v) for k, v in (frozen.get("session_at_run") or {}).items()},
    }


def _raise_on_exception(at: Any, step: str) -> None:
    if at.exception:
        details = "\n".join(str(e.value) + "\n" + "\n".join(e.stack_trace or []) for e in at.exception)
        raise AssertionError(f"page 9 raised during {step}:\n{details}")


# --------------------------------------------------------------------------------------------- the record
def _record(at: Any, namespace: Mapping[str, Any]) -> dict[str, Any]:
    from utils.bmo.snapshot import encode

    state = at.session_state
    # The full encoded results are in the snapshot (``snapshot.results``); this is the plan as an operator reads it.
    plan = {key: _plan(state[f"{PREFIX}{key}"]) for key in ("lp_result", "manual_blend") if f"{PREFIX}{key}" in state}
    plan.update(
        {
            key: encode(state[f"{PREFIX}{key}"])
            for key in ("lp_errors", "coke_anchor_basis", "coke_anchor_rate_kg_thm")
            if f"{PREFIX}{key}" in state
        }
    )
    payload = namespace.get("_charged_panel_payload") or {}
    charged = {
        "prediction": encode(payload.get("prediction")),
        "detail": encode(payload.get("detail")),
        "recent": encode(payload.get("recent")),
    }
    si = {
        "record": encode(namespace.get("si_furnace_forecast")),
        "tile": encode(namespace.get("live_si_result_tile")),
        "live_si_pct": encode(namespace.get("live_si_pct")),
    }
    return {
        "plan": plan,
        "context": {
            name: encode(namespace.get(name))
            for name in (
                "target_fe_mt",
                "target_slag_qty_mt",
                "max_burden_qty_mt",
                "observed_slag_rate",
                "fuel_rate_anchor_basis",
                "shift_baseline",
            )
        },
        "charged": charged,
        "si": si,
        "tables": _result_elements(at),
    }


def _plan(blend: Any) -> dict[str, Any] | None:
    """A blend evaluation without its model internals: shares, tonnes, cost, slag, fuel rates, limits."""

    if blend is None:
        return None
    diagnostics = blend.diagnostics or {}
    return {
        "shares_pct": dict(blend.shares_pct),
        "quantities_mt": dict(blend.quantities_mt),
        "total_qty_mt": blend.total_qty_mt,
        "ore_cost_per_thm_rs": blend.ore_cost_per_thm_rs,
        "fuel_cost_per_thm_rs": blend.fuel_cost_per_thm_rs,
        "objective_rs_per_thm": blend.objective_rs_per_thm,
        "fe_t_pct": blend.fe_t_pct,
        "slag_mt": blend.slag_mt,
        "slag_rate_kg_per_thm": blend.slag_rate_kg_per_thm,
        "slag_basicity": blend.slag_basicity,
        "slag_t_basicity": blend.slag_t_basicity,
        "slag_al2o3_pct": blend.slag_al2o3_pct,
        "slag_mgo_pct": blend.slag_mgo_pct,
        "feasible": blend.feasible,
        "violations": list(blend.violations),
        "fuel_rate_estimate": diagnostics.get("fuel_rate_estimate"),
        "fuel_rate_estimate_source": diagnostics.get("fuel_rate_estimate_source"),
        "coke_correction_delta_kg_thm": diagnostics.get("coke_correction_delta_kg_thm"),
        "flux_cost_per_thm_rs": diagnostics.get("flux_cost_per_thm_rs"),
        "lp_flux_quantities_mt": diagnostics.get("lp_flux_quantities_mt"),
        "adjusted_objective_rs_per_thm": diagnostics.get("adjusted_objective_rs_per_thm"),
    }


def _snapshot_record(folder: Path, scrub: Any) -> dict[str, Any]:
    files = sorted(folder.glob("*.json"))
    if len(files) != 1:
        raise AssertionError(f"expected one snapshot file, found {[f.name for f in files]}")
    snapshot = scrub(json.loads(files[0].read_text(encoding="utf-8")))
    frozen = snapshot.get("frozen") or {}
    return {
        "schema": snapshot.get("schema"),
        "source": snapshot.get("source"),
        "run": {k: v for k, v in (snapshot.get("run") or {}).items() if k != "at"},
        "inputs": snapshot.get("inputs"),
        "results": snapshot.get("results"),
        "context": snapshot.get("context"),
        "summary": snapshot.get("summary"),
        "not_restorable": snapshot.get("not_restorable"),
        "frozen": {
            "provider_calls": {k: _digest(v) for k, v in sorted((frozen.get("provider_calls") or {}).items())},
            "page": {k: _digest(v) for k, v in sorted((frozen.get("page") or {}).items())},
            "session_at_run": frozen.get("session_at_run"),
            "history_checks": frozen.get("history_checks"),
            "model_check": frozen.get("model_check"),
        },
    }


def _digest(value: Any) -> str:
    text = json.dumps(_rounded(value), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _rounded(value: Any) -> Any:
    """Floats at 9 significant digits, so a digest survives the last-bit noise of another BLAS."""

    if isinstance(value, float):
        return float(f"{value:.9g}") if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(k): _rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    return value


_RESULT_TABS = ("Balanced Optimizer", "Intensive Optimizer", "Comparison", "Model accuracy")
_RESULT_END = "### 📸 Snapshots"
_SKIP_END = "### Diagnostics and assumptions"


def _result_elements(at: Any) -> list[Any]:
    """The result section of the page, in order: from the result tabs to the diagnostics heading."""

    nodes = list(_walk(at.main))
    start = next(
        (i for i, n in enumerate(nodes) if n.type == "tab_container" and _tab_labels(n) == list(_RESULT_TABS)),
        None,
    )
    if start is None:
        return []
    out: list[Any] = []
    for node in nodes[start:]:
        text = _text(node)
        if text in (_SKIP_END, _RESULT_END):
            break
        item = _element(node)
        if item is not None:
            out.append(item)
    return out


def _walk(block: Any) -> Iterator[Any]:
    for child in getattr(block, "children", {}).values():
        yield child
        yield from _walk(child)


def _tab_labels(node: Any) -> list[str]:
    return [getattr(child, "label", "") for child in getattr(node, "children", {}).values()]


_TEXT_TYPES = ("markdown", "caption", "text", "latex", "code", "json")
_ALERT_TYPES = ("error", "warning", "info", "success", "exception")


def _text(node: Any) -> str | None:
    """The text of a text or alert element (never a widget's value: that would read session state)."""

    if node.type not in _TEXT_TYPES + _ALERT_TYPES:
        return None
    proto = getattr(node, "proto", None)
    body = getattr(proto, "body", None)
    if isinstance(body, str):
        return body
    value = getattr(node, "value", None)
    return value if isinstance(value, str) else None


def _element(node: Any) -> Any:
    kind = node.type
    if kind in _TEXT_TYPES + _ALERT_TYPES:
        return [kind, _text(node)]
    if kind == "metric":
        return ["metric", node.label, node.value, getattr(node.proto, "delta", ""), getattr(node.proto, "help", "")]
    if kind in ("dataframe", "table", "arrow_data_frame"):
        frame = node.value
        return ["table", [str(c) for c in frame.columns], _rows(frame)]
    if kind == "plotly_chart":
        return ["chart", _chart(node.proto.spec)]
    if kind in ("tab", "expander", "popover", "status"):
        return [kind, getattr(node, "label", None) or getattr(node.proto, "label", None)]
    if kind in ("button", "number_input", "checkbox", "toggle", "selectbox", "radio", "text_input", "data_editor"):
        return [kind, getattr(node.proto, "label", None)]
    if kind in (
        "tab_container",
        "vertical",
        "horizontal",
        "column",
        "form",
        "container",
        "divider",
        "empty",
        "flex_container",
    ):
        return None
    proto = getattr(node, "proto", None)
    body = getattr(proto, "body", None) if proto is not None else None
    return [kind, body] if isinstance(body, str) and body else [kind]


def _rows(frame: pandas.DataFrame) -> list[list[Any]]:
    return [[_cell(v) for v in row] for row in frame.itertuples(index=False, name=None)]


def _cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (pandas.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _chart(spec: str) -> list[dict[str, Any]]:
    """Each trace's name, type and points (count, min, max, sum of y): enough to see a moved series."""

    figure = json.loads(spec)
    traces = []
    for trace in figure.get("data", []):
        xs = _array(trace.get("x")) or _array(trace.get("labels"))
        ys = [v for v in (_array(trace.get("y")) or _array(trace.get("values"))) if _finite(v)]
        traces.append(
            {
                "type": trace.get("type"),
                "name": trace.get("name"),
                "points": len(xs) or len(ys),
                "x": [xs[0], xs[-1]] if xs else None,
                "y": [min(ys), max(ys), math.fsum(ys)] if ys else None,
            }
        )
    return traces


def _array(value: Any) -> list[Any]:
    """A Plotly data array: a list, or a typed array (``{"dtype", "bdata"}``, Plotly 6)."""

    if isinstance(value, dict) and "bdata" in value:
        import base64

        import numpy as np

        return np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"]).tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# --------------------------------------------------------------------------------------------- comparison
def compare(expected: Any, actual: Any, path: str = "$") -> list[str]:
    """Differences between two records: floats within ``RTOL`` (relative) or ``ATOL``, everything else exact."""

    if isinstance(expected, float) or isinstance(actual, float):
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and not isinstance(expected, bool):
            if math.isclose(float(expected), float(actual), rel_tol=RTOL, abs_tol=ATOL):
                return []
        return [f"{path}: {expected!r} != {actual!r}"]
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = [f"{path}.{k}: missing" for k in expected if k not in actual]
        out += [f"{path}.{k}: unexpected" for k in actual if k not in expected]
        for key in expected:
            if key in actual:
                out += compare(expected[key], actual[key], f"{path}.{key}")
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: length {len(expected)} != {len(actual)}"]
        out = []
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            out += compare(e, a, f"{path}[{i}]")
        return out
    return [] if expected == actual else [f"{path}: {expected!r} != {actual!r}"]


def normalise(record: Any) -> Any:
    """The record as stored: plain JSON (tuples as lists)."""

    return json.loads(json.dumps(record, ensure_ascii=False, default=str))


def write(case: Case, record: Mapping[str, Any]) -> Path:
    path = golden_path(case)
    text = json.dumps(normalise(record), indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    with gzip.GzipFile(path, "wb", mtime=0) as handle:
        handle.write(text.encode("utf-8"))
    return path


def build_plant_fixtures(snapshot_dir: Path = SRC / "storage" / "bmo_snapshots") -> list[Path]:
    """``plant_<id>.json.gz`` from the committed snapshots: their inputs and frozen plant data, not their results."""

    written = []
    for name in sorted({case.plant for case in CASES}):
        snapshot = json.loads((snapshot_dir / f"{name}_blend_mix_optimiser.json").read_text(encoding="utf-8"))
        frozen = dict(snapshot.get("frozen") or {})
        page = {k: v for k, v in (frozen.get("page") or {}).items() if k != "bmo_cfg"}
        plant = {
            "source_snapshot": snapshot.get("id"),
            "created_at": snapshot.get("created_at"),
            "run": snapshot.get("run"),
            "label": snapshot.get("label"),
            "inputs": snapshot.get("inputs"),
            "frozen": {
                "provider_calls": frozen.get("provider_calls"),
                "page": page,
                "session_at_run": frozen.get("session_at_run"),
            },
        }
        path = FIXTURES / f"plant_{name}.json.gz"
        FIXTURES.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(path, "wb", mtime=0) as handle:
            handle.write(json.dumps(plant, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        written.append(path)
    return written


def build_config_fixture() -> Path:
    """``setting_bmo.json``: the ``bmo`` block of ``src/config/setting_bmo.yml`` as the golden runs read it."""

    import yaml

    cfg = yaml.safe_load((SRC / "config" / "setting_bmo.yml").read_text(encoding="utf-8"))["bmo"]
    path = FIXTURES / "setting_bmo.json"
    FIXTURES.mkdir(parents=True, exist_ok=True)
    text = json.dumps(cfg, indent=1, ensure_ascii=False) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="rewrite the golden files")
    parser.add_argument("--day", action="append", help="only these days (default: all)")
    parser.add_argument("--inputs", action="store_true", help="rebuild the plant and config fixtures first")
    args = parser.parse_args(argv)
    for name, value in TEST_ENVIRONMENT.items():  # as tests/conftest.py, so a written record is the test's
        os.environ.setdefault(name, value)
    for path in (SRC, REPO / "furnace_data", REPO):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if args.inputs:
        for path in [*build_plant_fixtures(), build_config_fixture()]:
            print(f"input fixture: {path.relative_to(REPO)}")
    failures = 0
    for case in CASES:
        if args.day and case.day not in args.day:
            continue
        record = normalise(run_case(case))
        if args.write:
            print(f"{case.day}: written {write(case, record)}")
            continue
        diffs = compare(load_golden(case), record)
        failures += bool(diffs)
        print(f"{case.day}: {'unchanged' if not diffs else f'{len(diffs)} differences'}")
        for line in diffs[:40]:
            print("   ", line)
    return 1 if failures else 0


if __name__ == "__main__":
    # The AppTest script imports ``tests.bmo_golden``; run that module's ``main`` so both share ``CURRENT``.
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tests import bmo_golden as _module

    raise SystemExit(_module.main())
