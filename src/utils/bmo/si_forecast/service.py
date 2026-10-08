"""Issue the versioned live BF2 furnace-operation Si forecast path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data.bmo.si_forecast_context import SiliconForecastSources
from utils.bmo.si_forecast.features import (
    PLANT_TZ,
    SiliconFeatureError,
    build_features,
    matched_outcomes,
    prepare_labs,
)
from utils.bmo.si_forecast.ledger import SiliconForecastLedger
from utils.bmo.si_forecast.model import SiliconExtraTreesPredictor

LEGACY_MODE = "legacy"
SHADOW_MODE = "shadow"
FORECAST_MODE = "forecast"
MODES = frozenset({LEGACY_MODE, SHADOW_MODE, FORECAST_MODE})
SCHEMA_VERSION = "bf2-si-forecast/2"
MAX_MISSING_FEATURE_FRACTION = 0.25
PREDICTABLE_STATUSES = frozenset({"ok", "warning", "out_of_domain"})


@dataclass(frozen=True)
class SiliconReplayResult:
    """Read-only causal replay used to make a new live ledger interpretable."""

    forecasts: pd.DataFrame
    actuals: pd.DataFrame
    scored: pd.DataFrame


def _aware(value: Any, timezone: str = PLANT_TZ) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(timezone)
    return stamp


def forecast_origin(now: Any, *, cadence_minutes: int = 5) -> pd.Timestamp:
    """Floor in IST first, then return the storage clock in UTC."""

    return (
        _aware(now)
        .tz_convert(PLANT_TZ)
        .floor(f"{int(cadence_minutes)}min")
        .tz_convert("UTC")
    )


def _plant_to_utc(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(PLANT_TZ)
    return stamp.tz_convert("UTC").isoformat()


def _frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=True, date_format="%Y-%m-%dT%H:%M:%S.%f%z")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _feature_hash(row: pd.Series) -> str:
    payload = [None if not np.isfinite(float(value)) else float(value) for value in row]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _watermark(frame: pd.DataFrame, column: str) -> str | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return values.max().isoformat() if values.notna().any() else None


def _quality_status(row: pd.Series) -> tuple[str, list[str], list[str], bool]:
    """Return status, exact warnings and whether advisory inference is allowed."""

    if pd.isna(row.get("si_age_h")):
        return (
            "stale_inputs",
            ["missing_raw_si"],
            ["No earlier raw hot-metal Si sample is available."],
            False,
        )

    codes: list[str] = []
    reasons: list[str] = []

    def warn(code: str, message: str) -> None:
        codes.append(code)
        reasons.append(message)

    if not bool(row.get("gate_row_present", False)):
        warn("missing_gate_row", "The previous completed-hour operating record is missing.")
    gate_values = (
        "hot_blast_nm3h",
        "pci_kg_thm",
        "production_tph",
        "gas_total_pct",
        "fuel_rate_kg_thm",
    )
    missing_gate = [name for name in gate_values if pd.isna(row.get(name))]
    if missing_gate:
        warn(
            "missing_gate_value",
            "Previous-hour operating checks are missing: "
            + ", ".join(name.replace("_", " ") for name in missing_gate)
            + ".",
        )
    if float(row["si_age_h"]) > 12:
        warn(
            "stale_raw_si",
            f"The latest raw Si sample is {float(row['si_age_h']):.1f} hours old.",
        )
    if not bool(row.get("latest_flow_present", False)):
        warn(
            "latest_blast_flow_missing",
            "The latest blast-volume reading is missing.",
        )
    coverage = row.get("online_6h_fraction")
    if pd.isna(coverage):
        warn(
            "online_history_short",
            "Six hours of blast-volume history are not available.",
        )
    elif float(coverage) < 0.8:
        warn(
            "online_coverage_low",
            f"Only {float(coverage):.0%} of the required six-hour blast-volume "
            "history is available.",
        )

    checks = (
        (
            "hot_blast_ok",
            "hot_blast_low",
            lambda: f"Hot-blast volume is {float(row['hot_blast_nm3h']):,.0f} Nm3/h; model range starts at 90,000.",
        ),
        (
            "pci_ok",
            "pci_low",
            lambda: f"PCI is {float(row['pci_kg_thm']):,.1f} kg/THM; model range starts at 100.",
        ),
        (
            "production_ok",
            "production_low",
            lambda: f"Production is {float(row['production_tph']):,.1f} t/h; model range starts at 75.",
        ),
        (
            "gas_ok",
            "top_gas_outside",
            lambda: f"CO + CO2 is {float(row['gas_total_pct']):,.1f}%; model range is 38-47%.",
        ),
        (
            "fuel_ok",
            "fuel_rate_outside",
            lambda: f"Fuel rate is {float(row['fuel_rate_kg_thm']):,.1f} kg/THM; model range is 500-600.",
        ),
    )
    if not missing_gate:
        for flag, code, message in checks:
            if not bool(row.get(flag, False)):
                warn(code, message())
    return ("warning" if codes else "ok", codes, reasons, True)


def forecast_horizons(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize v1 and v2 records to one path contract."""

    path = [dict(row) for row in record.get("horizons") or []]
    if path:
        return sorted(path, key=lambda row: int(row.get("horizon_minutes", 0)))
    if not record.get("target_start") or not record.get("target_end"):
        return []
    origin = _aware(record.get("origin_at")).tz_convert("UTC")
    start = _aware(record["target_start"]).tz_convert("UTC")
    minutes = int(round((start - origin).total_seconds() / 60.0))
    interval = record.get("interval") or {}
    return [
        {
            "horizon_minutes": minutes,
            "label": "Now" if minutes == 0 else f"+{minutes / 60:g} h",
            "target_start": start.isoformat(),
            "target_end": _aware(record["target_end"]).tz_convert("UTC").isoformat(),
            "si_pct": record.get("si_pct"),
            "lower_pct": interval.get("lower_pct") if isinstance(interval, Mapping) else None,
            "upper_pct": interval.get("upper_pct") if isinstance(interval, Mapping) else None,
            "status": record.get("status"),
            "legacy": True,
        }
    ]


class SiliconForecastService:
    """Issue one immutable advisory path per model-aligned origin."""

    def __init__(
        self,
        *,
        bundle_dir: str | Path,
        storage_dir: str | Path,
        furnace_id: str = "BF2",
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.storage_dir = Path(storage_dir)
        self.furnace_id = str(furnace_id)
        self.ledger = SiliconForecastLedger(self.storage_dir)
        self.predictor: SiliconExtraTreesPredictor | None = None
        self.model_error: str | None = None
        try:
            self.predictor = SiliconExtraTreesPredictor(self.bundle_dir)
        except Exception as exc:  # noqa: BLE001 - failure becomes a forecast status
            self.model_error = str(exc)

    def _specifications(self, origin: pd.Timestamp) -> list[dict[str, Any]]:
        if self.predictor is None:
            return [
                {
                    "horizon_minutes": minutes,
                    "target_window_minutes": 5,
                }
                for minutes in (0, 60, 120, 180)
            ]
        return [
            {
                "horizon_minutes": minutes,
                "target_window_minutes": self.predictor.target_window_minutes(minutes),
            }
            for minutes in self.predictor.horizons
        ]

    def _unavailable(
        self, *, origin: pd.Timestamp, issued_at: pd.Timestamp, message: str
    ) -> dict[str, Any]:
        path = []
        for spec in self._specifications(origin):
            start = origin + pd.Timedelta(minutes=spec["horizon_minutes"])
            path.append(
                {
                    **spec,
                    "label": (
                        "Now"
                        if spec["horizon_minutes"] == 0
                        else f"+{spec['horizon_minutes'] / 60:g} h"
                    ),
                    "target_start": start.isoformat(),
                    "target_end": (
                        start + pd.Timedelta(minutes=spec["target_window_minutes"])
                    ).isoformat(),
                    "si_pct": None,
                    "lower_pct": None,
                    "upper_pct": None,
                    "status": "model_unavailable",
                }
            )
        headline = path[0]
        return {
            "schema_version": SCHEMA_VERSION,
            "furnace_id": self.furnace_id,
            "status": "model_unavailable",
            "reason_codes": ["model_unavailable"],
            "reasons": [message],
            "si_pct": None,
            "origin_at": origin.isoformat(),
            "issued_at": issued_at.isoformat(),
            "target_start": headline["target_start"],
            "target_end": headline["target_end"],
            "interval": None,
            "horizons": path,
        }

    def record_matured_outcomes(
        self, sources: SiliconForecastSources, *, available_by: pd.Timestamp
    ) -> int:
        recorded = 0
        for origin, issue in self.ledger.issues().items():
            if issue.get("status") not in PREDICTABLE_STATUSES:
                continue
            for horizon in forecast_horizons(issue):
                if horizon.get("si_pct") is None:
                    continue
                target_end = _aware(horizon["target_end"]).tz_convert("UTC")
                if target_end > available_by.tz_convert("UTC"):
                    continue
                matches = matched_outcomes(
                    sources.labs,
                    target_start=_aware(horizon["target_start"]),
                    target_end=target_end,
                    available_by=available_by,
                )
                for row in matches.itertuples():
                    sample_at = (
                        pd.Timestamp(row.sample).tz_localize(PLANT_TZ).tz_convert("UTC")
                    )
                    available_at = (
                        pd.Timestamp(row.available).tz_localize(PLANT_TZ).tz_convert("UTC")
                    )
                    recorded += int(
                        self.ledger.record_outcome(
                            origin_at=origin,
                            horizon_minutes=int(horizon["horizon_minutes"]),
                            sample_id=str(row.id),
                            lab_sample_id=str(row.lab_sample_id),
                            sample_at=sample_at,
                            available_at=available_at,
                            actual_si_pct=float(row.chem_pct_si),
                            recorded_at=available_by,
                        )
                    )
        return recorded

    def replay_history(
        self,
        sources: SiliconForecastSources,
        *,
        origin_start: Any,
        origin_end: Any,
        available_by: Any | None = None,
    ) -> SiliconReplayResult:
        """Run the active bundle over past issue clocks without writing the ledger.

        Every historical row is rebuilt with the same causal clocks as production:
        a lab or charge is usable only when both its event time and database
        availability precede that row's simulated issue time.  The result is a
        replay, not an as-issued production record, and callers must label it so.
        """

        forecast_columns = [
            "origin_at",
            "issued_at",
            "horizon_minutes",
            "target_start",
            "target_end",
            "prediction",
            "lower",
            "upper",
            "status",
            "warnings",
            "model_version",
            "last_sample_si_pct",
            "history_kind",
        ]
        actual_columns = ["sample_id", "sample_at", "actual", "available_at"]
        scored_columns = [
            "origin_at",
            "sample_at",
            "sample_id",
            "horizon_minutes",
            "target_start",
            "target_end",
            "actual",
            "prediction",
            "persistence",
            "lower",
            "upper",
            "inside_range",
            "status",
            "model_version",
            "history_kind",
        ]

        def empty() -> SiliconReplayResult:
            return SiliconReplayResult(
                pd.DataFrame(columns=forecast_columns),
                pd.DataFrame(columns=actual_columns),
                pd.DataFrame(columns=scored_columns),
            )

        if self.predictor is None:
            return empty()

        cadence = int(self.predictor.cadence_minutes)
        frequency = f"{cadence}min"
        start = _aware(origin_start).tz_convert(PLANT_TZ).ceil(frequency)
        end = _aware(origin_end).tz_convert(PLANT_TZ).floor(frequency)
        if end < start:
            return empty()
        origins = pd.date_range(start, end, freq=frequency)
        built = build_features(
            sources.online,
            sources.labs,
            sources.charge,
            sources.hourly_gate,
            origins,
            self.bundle_dir,
        )
        predicted = self.predictor.predict_horizons(built.features)
        ordered = built.features.loc[:, self.predictor.features]

        forecast_rows: list[dict[str, Any]] = []
        for position, origin in enumerate(built.features.index):
            feature_row = built.features.iloc[position]
            quality = built.quality.iloc[position]
            status, codes, reasons, can_predict = _quality_status(quality)
            diagnostics = [
                self.predictor.input_diagnostics(
                    ordered.iloc[[position]].to_numpy(dtype=float), horizon
                )
                for horizon in self.predictor.horizons
            ]
            missing_fraction = max(
                row["missing_feature_fraction"] for row in diagnostics
            )
            if missing_fraction > MAX_MISSING_FEATURE_FRACTION:
                can_predict = False
                status = "insufficient_data"
                codes = [*codes, "model_feature_coverage_low"]
                reasons = [
                    *reasons,
                    f"{missing_fraction:.0%} of model inputs are missing; "
                    f"the limit is {MAX_MISSING_FEATURE_FRACTION:.0%}.",
                ]
            origin_utc = pd.Timestamp(origin).tz_convert("UTC")
            persistence = (
                float(feature_row["si_last"])
                if pd.notna(feature_row["si_last"])
                else None
            )
            for horizon in self.predictor.horizons:
                target_start = origin_utc + pd.Timedelta(minutes=horizon)
                width = self.predictor.target_window_minutes(horizon)
                value = float(predicted[horizon][position]) if can_predict else None
                interval = self.predictor.interval_half_width(horizon)
                forecast_rows.append(
                    {
                        "origin_at": origin_utc,
                        "issued_at": origin_utc,
                        "horizon_minutes": int(horizon),
                        "target_start": target_start,
                        "target_end": target_start + pd.Timedelta(minutes=width),
                        "prediction": value,
                        "lower": (
                            max(0.0, value - interval)
                            if value is not None and interval is not None
                            else None
                        ),
                        "upper": (
                            value + interval
                            if value is not None and interval is not None
                            else None
                        ),
                        "status": status,
                        "warnings": reasons,
                        "reason_codes": codes,
                        "model_version": self.predictor.version,
                        "last_sample_si_pct": persistence,
                        "history_kind": "causal_replay",
                    }
                )
        forecasts = pd.DataFrame(forecast_rows, columns=[*forecast_columns, "reason_codes"])

        seen_value = sources.fetched_at if available_by is None else available_by
        seen = _aware(seen_value).tz_convert(PLANT_TZ)
        seen_naive = seen.tz_localize(None)
        lab_rows = prepare_labs(sources.labs)
        lab_rows = lab_rows[
            lab_rows["available"].le(seen_naive)
            & lab_rows["sample"].ge(start.tz_localize(None))
            & lab_rows["sample"].le(seen_naive)
        ].copy()
        actuals = pd.DataFrame(
            {
                "sample_id": lab_rows["id"].astype(str).to_numpy(),
                "sample_at": pd.DatetimeIndex(lab_rows["sample"])
                .tz_localize(PLANT_TZ)
                .tz_convert("UTC"),
                "actual": lab_rows["chem_pct_si"].to_numpy(dtype=float),
                "available_at": pd.DatetimeIndex(lab_rows["available"])
                .tz_localize(PLANT_TZ)
                .tz_convert("UTC"),
            },
            columns=actual_columns,
        )

        scored_rows: list[dict[str, Any]] = []
        for row in forecasts.itertuples(index=False):
            if row.prediction is None or not np.isfinite(float(row.prediction)):
                continue
            matches = actuals[
                actuals["sample_at"].ge(row.target_start)
                & actuals["sample_at"].lt(row.target_end)
            ]
            for actual in matches.itertuples(index=False):
                inside = (
                    bool(float(row.lower) <= float(actual.actual) <= float(row.upper))
                    if row.lower is not None and row.upper is not None
                    else None
                )
                scored_rows.append(
                    {
                        "origin_at": row.origin_at,
                        "sample_at": actual.sample_at,
                        "sample_id": actual.sample_id,
                        "horizon_minutes": int(row.horizon_minutes),
                        "target_start": row.target_start,
                        "target_end": row.target_end,
                        "actual": float(actual.actual),
                        "prediction": float(row.prediction),
                        "persistence": row.last_sample_si_pct,
                        "lower": row.lower,
                        "upper": row.upper,
                        "inside_range": inside,
                        "status": row.status,
                        "model_version": row.model_version,
                        "history_kind": "causal_replay",
                    }
                )
        scored = pd.DataFrame(scored_rows, columns=scored_columns)
        return SiliconReplayResult(forecasts, actuals, scored)

    def issue(
        self,
        sources: SiliconForecastSources,
        *,
        origin: Any,
        issued_at: Any | None = None,
        mode: str = SHADOW_MODE,
    ) -> dict[str, Any]:
        if mode not in MODES:
            raise ValueError(f"Unknown silicon forecast mode: {mode}")
        cadence = self.predictor.cadence_minutes if self.predictor else 5
        origin_utc = forecast_origin(origin, cadence_minutes=cadence)
        issued = _aware(issued_at or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")
        existing = self.ledger.get(origin_utc)
        if existing is not None:
            self.record_matured_outcomes(sources, available_by=issued)
            return existing
        if self.predictor is None:
            return self._unavailable(
                origin=origin_utc,
                issued_at=issued,
                message=self.model_error or "The active silicon model could not be loaded.",
            )
        try:
            built = build_features(
                sources.online,
                sources.labs,
                sources.charge,
                sources.hourly_gate,
                [origin_utc],
                self.bundle_dir,
            )
        except (SiliconFeatureError, OSError, ValueError, KeyError) as exc:
            unavailable = self._unavailable(
                origin=origin_utc, issued_at=issued, message=str(exc)
            )
            unavailable["status"] = "insufficient_data"
            unavailable["reason_codes"] = [getattr(exc, "code", "feature_build_failed")]
            for row in unavailable["horizons"]:
                row["status"] = "insufficient_data"
            return unavailable

        feature_row = built.features.iloc[0]
        quality = built.quality.iloc[0]
        status, codes, reasons, can_predict = _quality_status(quality)
        diagnostics = [
            self.predictor.input_diagnostics(
                built.features.loc[:, self.predictor.features].to_numpy(dtype=float),
                horizon,
            )
            for horizon in self.predictor.horizons
        ]
        input_diag = {
            "missing_feature_fraction": max(
                row["missing_feature_fraction"] for row in diagnostics
            ),
            "clipped_feature_fraction": max(
                row["clipped_feature_fraction"] for row in diagnostics
            ),
            "missing_features": sorted(
                {name for row in diagnostics for name in row.get("missing_features", [])}
            ),
        }
        if input_diag["missing_feature_fraction"] > MAX_MISSING_FEATURE_FRACTION:
            can_predict = False
            status = "insufficient_data"
            codes.append("model_feature_coverage_low")
            reasons.append(
                f"{input_diag['missing_feature_fraction']:.0%} of model inputs are missing; "
                f"the limit is {MAX_MISSING_FEATURE_FRACTION:.0%}."
            )

        predictions = (
            self.predictor.predict_horizons(built.features) if can_predict else {}
        )
        path: list[dict[str, Any]] = []
        for horizon in self.predictor.horizons:
            start = origin_utc + pd.Timedelta(minutes=horizon)
            width = self.predictor.target_window_minutes(horizon)
            value = float(predictions[horizon][0]) if can_predict else None
            interval = self.predictor.interval_half_width(horizon)
            lower = max(0.0, value - interval) if value is not None and interval is not None else None
            upper = value + interval if value is not None and interval is not None else None
            path.append(
                {
                    "horizon_minutes": horizon,
                    "label": "Now" if horizon == 0 else f"+{horizon / 60:g} h",
                    "target_start": start.isoformat(),
                    "target_end": (start + pd.Timedelta(minutes=width)).isoformat(),
                    "si_pct": value,
                    "lower_pct": lower,
                    "upper_pct": upper,
                    "status": status,
                    "legacy": not self.predictor.is_multi_horizon,
                }
            )
        headline = next((row for row in path if row["horizon_minutes"] == 0), path[0])
        online_last = _watermark(sources.online, "time (IST)")
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "furnace_id": self.furnace_id,
            "mode": mode,
            "model_id": self.predictor.model_id,
            "model_version": self.predictor.version,
            "model_bundle": str(self.bundle_dir),
            "status": status,
            "reason_codes": codes,
            "reasons": reasons,
            "si_pct": headline["si_pct"],
            "origin_at": origin_utc.isoformat(),
            "issued_at": issued.isoformat(),
            "target_start": headline["target_start"],
            "target_end": headline["target_end"],
            "interval": (
                {
                    "lower_pct": headline["lower_pct"],
                    "upper_pct": headline["upper_pct"],
                    "coverage": 0.90,
                }
                if headline["lower_pct"] is not None
                else None
            ),
            "horizons": path,
            "last_sample_at": _plant_to_utc(quality.get("last_sample_at")),
            "last_sample_available_at": _plant_to_utc(
                quality.get("last_sample_available_at")
            ),
            "last_sample_si_pct": (
                float(feature_row["si_last"])
                if pd.notna(feature_row["si_last"])
                else None
            ),
            "online_last_bin_end": online_last,
            "coverage": {
                "blast_flow_6h_fraction": (
                    float(quality["online_6h_fraction"])
                    if pd.notna(quality["online_6h_fraction"])
                    else None
                )
            },
            **input_diag,
            "input_hash": _feature_hash(feature_row),
            "source_hashes": {
                "online": _frame_hash(sources.online),
                "labs": _frame_hash(sources.labs),
                "charge": _frame_hash(sources.charge),
                "hourly_gate": _frame_hash(sources.hourly_gate),
            },
            "source_watermarks": {
                "online": online_last,
                "lab_sample": _watermark(sources.labs, "time (IST)"),
                "lab_created": _watermark(sources.labs, "created_at"),
                "charge_event": _watermark(sources.charge, "time (IST)"),
                "charge_created": _watermark(sources.charge, "created_at"),
                "hourly_gate": _watermark(sources.hourly_gate, "time"),
            },
            "operating_gate": {
                name: (None if pd.isna(quality[name]) else float(quality[name]))
                for name in (
                    "hot_blast_nm3h",
                    "pci_kg_thm",
                    "production_tph",
                    "gas_total_pct",
                    "fuel_rate_kg_thm",
                )
            },
        }
        if can_predict:
            record = self.ledger.record_issue(record)
        self.record_matured_outcomes(sources, available_by=issued)
        return record


__all__ = [
    "FORECAST_MODE",
    "LEGACY_MODE",
    "MODES",
    "PREDICTABLE_STATUSES",
    "SCHEMA_VERSION",
    "SHADOW_MODE",
    "SiliconForecastService",
    "SiliconReplayResult",
    "forecast_horizons",
    "forecast_origin",
]
