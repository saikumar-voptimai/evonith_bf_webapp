"""Time-safe features and eligibility for the BF2 raw-cast Si forecast."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PLANT_TZ = "Asia/Kolkata"
HORIZON_MINUTES = (0, 60, 120, 180)
TARGET_WINDOW_MINUTES = 5

GATE_COLUMNS = (
    "PCI_KG/THM",
    "ACT. FUEL RATEKG/THM.",
    "PRODUCTIONTONNESPERHR",
    "HOT BLAST VOLUMENM3/HR.",
    "FURNACE TOP GAS ANALYSISCO2%",
    "FURNACE TOP GAS ANALYSISONLINE (ANALYZER)CO%",
)
BLAST_FLOW_COLUMN = "Process Params - BF2_PROC Hot Blast Volume"
LAB_COLUMNS = (
    "id", "lab_sample_id", "created_at", "cast_no_ladle_spec", "hmt_gt_1480c",
    "chem_pct_c", "chem_pct_cr", "chem_pct_fe", "chem_pct_mn", "chem_pct_p",
    "chem_pct_s", "chem_pct_si", "chem_pct_ti", "slag_basicity",
    "slag_pct_al2o3", "slag_pct_cao", "slag_pct_feo", "slag_pct_k2o",
    "slag_pct_mgo", "slag_pct_mno", "slag_pct_na2o", "slag_pct_s",
    "slag_pct_sio2", "slag_pct_tio2", "slag_t_basicity",
)
CHARGE_GROUPS = {
    "coke": ["coke_1_mt", "coke_2_mt"],
    "nut": ["nut_coke_1_mt", "nut_coke_2_mt"],
    "sinter": [f"sinter_{i}_mt" for i in range(1, 5)],
    "ore": [f"ore_{i}_mt" for i in range(1, 13)],
    "pellet": ["pellet_1_mt", "pellet_2_mt"],
}
CHARGE_COLUMNS = (
    "id", "created_at", "charge_no", "import_batch_id", "source_row_number",
    *[column for columns in CHARGE_GROUPS.values() for column in columns],
)


class SiliconFeatureError(ValueError):
    """A source-schema or clock violation that must not be silently imputed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SiliconFeatureResult:
    features: pd.DataFrame
    quality: pd.DataFrame


def _require(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(map(str, frame.columns)))
    if missing:
        raise SiliconFeatureError(
            "schema_error", f"{source} is missing required columns: {missing}"
        )


def ist_naive(values: pd.Series) -> pd.Series:
    """Convert explicitly zoned timestamps to naive plant-clock timestamps."""

    non_null = values.dropna()
    if any(pd.Timestamp(value).tzinfo is None for value in non_null):
        raise SiliconFeatureError(
            "naive_timestamp",
            "Source adapter supplied a timestamp without an explicit timezone.",
        )
    return (
        pd.to_datetime(values, utc=True, format="mixed", errors="coerce")
        .dt.tz_convert(PLANT_TZ)
        .dt.tz_localize(None)
    )


def _origins(
    values: Iterable[Any], *, cadence_minutes: int = 60
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    origins = pd.DatetimeIndex(values)
    if origins.tz is None:
        raise SiliconFeatureError(
            "naive_origin", "Forecast origins must be explicitly timezone-aware."
        )
    plant = origins.tz_convert(PLANT_TZ)
    cadence = f"{int(cadence_minutes)}min"
    if not (plant == plant.floor(cadence)).all():
        raise SiliconFeatureError(
            "origin_not_aligned",
            f"Forecast origins must be aligned to {cadence_minutes}-minute IST ticks.",
        )
    return plant, plant.tz_localize(None)


def prepare_labs(labs: pd.DataFrame) -> pd.DataFrame:
    """Clean raw assays with both sample and database-availability clocks."""

    _require(labs, ("time (IST)", *LAB_COLUMNS), "raw HM/slag feed")
    rows = labs.copy()
    rows["sample"] = ist_naive(rows["time (IST)"])
    rows["created"] = ist_naive(rows["created_at"])
    rows["available"] = rows[["sample", "created"]].max(axis=1)
    rows["chem_pct_si"] = pd.to_numeric(rows["chem_pct_si"], errors="coerce")
    rows = rows[
        rows["sample"].notna()
        & rows["created"].notna()
        & rows["chem_pct_si"].between(0.01, 3.0)
    ]
    rows = rows.sort_values(["sample", "available", "id"])
    chemistry = [
        column
        for column in labs.columns
        if str(column).startswith(("chem_", "slag_"))
    ] + ["hmt_gt_1480c"]
    return rows.drop_duplicates(["sample", "lab_sample_id", *chemistry])


def prepare_charges(charge: pd.DataFrame) -> pd.DataFrame:
    _require(charge, ("time (IST)", *CHARGE_COLUMNS), "raw charge feed")
    rows = charge.copy()
    rows["event"] = ist_naive(rows["time (IST)"])
    rows["created"] = ist_naive(rows["created_at"])
    rows["available"] = rows[["event", "created"]].max(axis=1)
    return (
        rows.dropna(subset=["event", "created"])
        .drop_duplicates("id")
        .sort_values("event")
    )


def matched_outcomes(
    labs: pd.DataFrame,
    *,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    available_by: pd.Timestamp,
) -> pd.DataFrame:
    """Eligible raw assay outcomes for one issued forecast target band."""

    rows = prepare_labs(labs)
    start = pd.Timestamp(target_start).tz_convert(PLANT_TZ).tz_localize(None)
    end = pd.Timestamp(target_end).tz_convert(PLANT_TZ).tz_localize(None)
    seen = pd.Timestamp(available_by).tz_convert(PLANT_TZ).tz_localize(None)
    rows = rows[
        rows["sample"].ge(start)
        & rows["sample"].lt(end)
        & rows["available"].le(seen)
    ]
    # A same-time conflict has no defensible single realised value. Exact repeat
    # records were already collapsed above; retain distinct samples otherwise.
    return rows[~rows["sample"].duplicated(False)].copy()


def build_features(
    online: pd.DataFrame,
    labs: pd.DataFrame,
    charge: pd.DataFrame,
    hourly_gate: pd.DataFrame,
    origins: Iterable[Any],
    bundle_dir: str | Path,
    *,
    clock_version: int | None = None,
) -> SiliconFeatureResult:
    """Build causal features on the artifact's hourly or five-minute clock.

    Format-v1 artifacts retain the exact published hourly feature construction.
    Format-v2 artifacts use the same 40 feature definitions on five-minute issue
    ticks, with ten-minute online bins and the inherited one-bin latency.
    """

    bundle = Path(bundle_dir)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    clock = int(
        clock_version
        or (2 if int(manifest.get("format_version", 1)) >= 2 else 1)
    )
    cadence_minutes = TARGET_WINDOW_MINUTES if clock >= 2 else 60
    plant_origins, ticks = _origins(origins, cadence_minutes=cadence_minutes)
    feature_specs = json.loads(
        (bundle / "selected_feature_manifest.json").read_text(encoding="utf-8")
    )
    channel_map = json.loads(
        (bundle / "online_channel_map.json").read_text(encoding="utf-8")
    )
    required_online = [row["source_display_column"] for row in channel_map]
    _require(online, ("time (IST)", *required_online), "ten-minute online feed")

    raw = online.loc[:, ["time (IST)", *required_online]].copy()
    raw.index = pd.DatetimeIndex(ist_naive(raw.pop("time (IST)")))
    if not raw.index.is_unique:
        raise SiliconFeatureError(
            "duplicate_online_bin", "Duplicate ten-minute online bin endpoints."
        )
    raw = raw.sort_index()
    raw = raw[raw.index == raw.index.floor("10min")]
    if raw.empty:
        raise SiliconFeatureError("no_online_bins", "No complete ten-minute bins.")
    # V1 needs the empty hourly-origin row used by the published replay. V2
    # selects an explicit completed-bin anchor for each five-minute origin.
    grid_end = max(raw.index.max(), ticks.max().ceil("10min"))
    raw = raw.reindex(pd.date_range(raw.index.min(), grid_end, freq="10min"))
    raw = raw.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for column in raw:
        if any(
            text in column
            for text in ("Temperature Profile", " Temp", "Delta T", "Heatload", "Cooling Water")
        ):
            raw[column] = raw[column].mask(raw[column] < 0)
        if column.startswith("Temperature Profile"):
            raw[column] = raw[column].mask(raw[column] == 0)

    values: dict[str, np.ndarray] = {}
    for spec in feature_specs:
        if spec["group"] != "online":
            continue
        source = str(spec["provenance"])
        if source not in raw:
            dependencies = list(spec["raw_online_dependencies"])
            _require(raw, dependencies, f"derived online channel {source}")
            components = raw[dependencies]
            raw[source] = (
                components.median(axis=1)
                if source.endswith("_median")
                else components.max(axis=1) - components.min(axis=1)
            )
        name = str(spec["feature"])
        if clock == 1:
            series = raw[source].resample(
                "h", offset="50min", closed="right", label="right"
            ).mean()
            series.index += pd.Timedelta(minutes=10)
            if "__mean_" in name:
                low, high = map(int, name.split("__mean_")[1][:-1].split("_"))
                engineered = series.shift(low).rolling(
                    high - low, min_periods=max(1, int((high - low) * 0.8))
                ).mean()
            else:
                engineered = series - series.shift(4)
            values[name] = engineered.reindex(ticks).to_numpy()
        else:
            anchors = ticks.floor("10min") - pd.Timedelta(minutes=10)
            series = raw[source]
            if "__mean_" in name:
                low, high = map(int, name.split("__mean_")[1][:-1].split("_"))
                width = (high - low) * 6
                engineered = series.shift(low * 6).rolling(
                    width, min_periods=max(1, int(width * 0.8))
                ).mean()
            else:
                engineered = series - series.shift(24)
            values[name] = engineered.reindex(anchors).to_numpy()

    lab_rows = prepare_labs(labs)
    lab_history: list[dict[str, Any]] = []
    for tick in ticks:
        past = lab_rows[
            lab_rows["sample"].lt(tick) & lab_rows["available"].le(tick)
        ]
        past = past[~past["sample"].duplicated(False)].tail(3)
        row: dict[str, Any] = {
            "si_last": np.nan,
            "si_age_h": np.nan,
            "si_prev2": np.nan,
            "si_mean3": np.nan,
            "last_sample_at": pd.NaT,
            "last_sample_available_at": pd.NaT,
        }
        if len(past):
            latest = past.iloc[-1]
            row.update(
                si_last=latest["chem_pct_si"],
                si_age_h=(tick - latest["sample"]).total_seconds() / 3600.0,
                si_prev2=(past.iloc[-2]["chem_pct_si"] if len(past) > 1 else np.nan),
                si_mean3=past["chem_pct_si"].mean(),
                last_sample_at=latest["sample"],
                last_sample_available_at=latest["available"],
            )
        lab_history.append(row)
    for name in ("si_last", "si_age_h", "si_prev2", "si_mean3"):
        values[name] = np.asarray([row[name] for row in lab_history], dtype=np.float32)

    charge_rows = prepare_charges(charge)
    mass = pd.DataFrame(
        {
            group: charge_rows[columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum(axis=1)
            for group, columns in CHARGE_GROUPS.items()
        }
    )
    ratio: list[float] = []
    pellet: list[float] = []
    for tick in ticks:
        chosen = (
            charge_rows["event"].ge(tick - pd.Timedelta(hours=3))
            & charge_rows["event"].lt(tick - pd.Timedelta(hours=1))
            & charge_rows["available"].le(tick)
        )
        if not chosen.any():
            ratio.append(np.nan)
            pellet.append(np.nan)
            continue
        summed = mass.loc[chosen].sum()
        fuel = summed["coke"] + summed["nut"]
        ratio.append(
            (summed["sinter"] + summed["ore"] + summed["pellet"]) / fuel
            if fuel > 0
            else np.nan
        )
        pellet.append(summed["pellet"] / 2.0)
    values["charge_1_3h_ore_coke_ratio"] = np.asarray(ratio, dtype=np.float32)
    values["charge_1_3h_pellet_mtph"] = np.asarray(pellet, dtype=np.float32)

    feature_order = list(manifest["feature_order"])
    missing_features = sorted(set(feature_order) - set(values))
    if missing_features:
        raise SiliconFeatureError(
            "feature_schema_error", f"Could not construct features: {missing_features}"
        )
    feature_frame = pd.DataFrame(values, index=plant_origins).loc[:, feature_order]
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)

    _require(hourly_gate, ("time", *GATE_COLUMNS), "hourly operating gate")
    gate = hourly_gate.copy()
    gate.index = pd.DatetimeIndex(ist_naive(gate.pop("time")))
    if not gate.index.is_unique:
        raise SiliconFeatureError(
            "duplicate_gate_row", "Duplicate hourly operating-gate records."
        )
    for column in GATE_COLUMNS:
        gate[column] = pd.to_numeric(gate[column], errors="coerce")
    lookup = (
        ticks - pd.Timedelta(hours=1)
        if clock == 1
        else ticks.floor("h") - pd.Timedelta(hours=1)
    )
    gate_present = np.asarray([timestamp in gate.index for timestamp in lookup])
    gate = gate.reindex(lookup)
    gate.index = plant_origins

    if clock == 1:
        flow = raw[BLAST_FLOW_COLUMN].shift(1).notna()
        flow_ticks = ticks
    else:
        flow = raw[BLAST_FLOW_COLUMN].notna()
        flow_ticks = ticks.floor("10min") - pd.Timedelta(minutes=10)
    coverage = flow.rolling(36, min_periods=36).mean().reindex(flow_ticks).to_numpy()
    latest = np.asarray(
        [bool(flow.get(tick, False)) for tick in flow_ticks], dtype=bool
    )
    gas = (
        gate["FURNACE TOP GAS ANALYSISCO2%"]
        + gate["FURNACE TOP GAS ANALYSISONLINE (ANALYZER)CO%"]
    )
    hot_blast_ok = gate["HOT BLAST VOLUMENM3/HR."].ge(90000)
    pci_ok = gate["PCI_KG/THM"].ge(100)
    production_ok = gate["PRODUCTIONTONNESPERHR"].ge(75)
    gas_ok = gas.between(38, 47)
    fuel_ok = gate["ACT. FUEL RATEKG/THM."].between(500, 600)
    broad = hot_blast_ok & pci_ok & production_ok & gas_ok & fuel_ok
    quality = pd.DataFrame(
        {
            "gate_row_present": gate_present,
            "broad_operating_gate": broad.to_numpy(dtype=bool),
            "hot_blast_ok": hot_blast_ok.to_numpy(dtype=bool),
            "pci_ok": pci_ok.to_numpy(dtype=bool),
            "production_ok": production_ok.to_numpy(dtype=bool),
            "gas_ok": gas_ok.to_numpy(dtype=bool),
            "fuel_ok": fuel_ok.to_numpy(dtype=bool),
            "hot_blast_nm3h": gate["HOT BLAST VOLUMENM3/HR."].to_numpy(),
            "pci_kg_thm": gate["PCI_KG/THM"].to_numpy(),
            "production_tph": gate["PRODUCTIONTONNESPERHR"].to_numpy(),
            "gas_total_pct": gas.to_numpy(),
            "fuel_rate_kg_thm": gate["ACT. FUEL RATEKG/THM."].to_numpy(),
            "online_6h_fraction": coverage,
            "latest_flow_present": latest,
            "si_age_h": feature_frame["si_age_h"].to_numpy(),
            "last_sample_at": [row["last_sample_at"] for row in lab_history],
            "last_sample_available_at": [
                row["last_sample_available_at"] for row in lab_history
            ],
        },
        index=plant_origins,
    )
    quality["eligible"] = (
        quality["gate_row_present"]
        & quality["broad_operating_gate"]
        & feature_frame["si_last"].notna()
        & feature_frame["si_age_h"].le(12)
        & quality["online_6h_fraction"].ge(0.8)
        & quality["latest_flow_present"]
    )
    return SiliconFeatureResult(feature_frame, quality)


def supervised_examples(
    result: SiliconFeatureResult,
    labs: pd.DataFrame,
    *,
    available_by: pd.Timestamp,
    horizon_minutes: int = 120,
    target_window_minutes: int = 60,
) -> pd.DataFrame:
    """Attach real raw-cast targets to one direct forecast horizon.

    The target is never interpolated. A sample is assigned to the issue tick at
    the start of its target bin, then shifted back by the requested horizon.
    """

    rows = prepare_labs(labs)
    seen = pd.Timestamp(available_by).tz_convert(PLANT_TZ).tz_localize(None)
    rows = rows[rows["available"].le(seen)]
    rows = rows[~rows["sample"].duplicated(False)].copy()
    rows["origin"] = rows["sample"].dt.floor(
        f"{int(target_window_minutes)}min"
    ) - pd.Timedelta(minutes=int(horizon_minutes))
    origin_ist = pd.DatetimeIndex(result.features.index).tz_convert(PLANT_TZ)
    lookup = {timestamp.tz_localize(None): timestamp for timestamp in origin_ist}
    rows = rows[rows["origin"].isin(lookup)].copy()
    rows["origin_aware"] = rows["origin"].map(lookup)
    eligible = result.quality["eligible"]
    rows = rows[rows["origin_aware"].map(eligible).fillna(False)].copy()
    if rows.empty:
        return pd.DataFrame()
    feature_rows = result.features.loc[rows["origin_aware"]].reset_index(drop=True)
    meta = pd.DataFrame(
        {
            "id": rows["id"].astype(str).to_numpy(),
            "lab_sample_id": rows["lab_sample_id"].astype(str).to_numpy(),
            "origin": rows["origin_aware"].to_numpy(),
            "horizon_minutes": int(horizon_minutes),
            "target_start": (
                rows["origin_aware"] + pd.Timedelta(minutes=int(horizon_minutes))
            ).to_numpy(),
            "target_end": (
                rows["origin_aware"]
                + pd.Timedelta(minutes=int(horizon_minutes + target_window_minutes))
            ).to_numpy(),
            "sample_time": rows["sample"].to_numpy(),
            "available": rows["available"].to_numpy(),
            "actual": rows["chem_pct_si"].to_numpy(dtype=float),
        }
    )
    return pd.concat([meta, feature_rows], axis=1)


__all__ = [
    "BLAST_FLOW_COLUMN",
    "CHARGE_COLUMNS",
    "GATE_COLUMNS",
    "HORIZON_MINUTES",
    "LAB_COLUMNS",
    "PLANT_TZ",
    "TARGET_WINDOW_MINUTES",
    "SiliconFeatureError",
    "SiliconFeatureResult",
    "build_features",
    "ist_naive",
    "matched_outcomes",
    "prepare_charges",
    "prepare_labs",
    "supervised_examples",
]
