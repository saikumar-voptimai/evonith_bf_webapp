"""Rolling bias correction for the energy balance's coke rate.

The calibration target is the plant's measured hourly coke rate: 1,000 times
``COKE_CALC_MT`` divided by hot-metal tonnes in the same hour, mass-weighted
across each complete day. A target-basis identifier is persisted with every
offset so a value fitted to an older definition can never be applied silently.

WHY AN OFFSET AND NOT A FITTED MODEL. The offset has one parameter, is directly
inspectable, and does not introduce feature coefficients that can fight the
physics it is correcting. It also shrinks to nothing as underlying balance
defects are fixed.

WHY ROLLING AND NOT CONSTANT. The shell-loss basis, burden and analyser state
all move over time. A 90-day window is long enough to average daily noise while
allowing the level correction to be refreshed as those conditions change.

WHAT THIS IS NOT. It is a bias correction, not a fix. When the analyser and
shell-loss questions come back from the plant, the offset should shrink toward
zero on its own - and if it does not, something else is wrong. Its size is a
standing measure of how much the balance is still missing, which is why it is
displayed rather than folded in silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

CALIBRATION_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "coke_rate_calibration.json"
)
DEFAULT_WINDOW_DAYS = 90
ACTUAL_COKE_TARGET_BASIS = "hourly_coke_calc_mt_per_hot_metal_v1"
LEGACY_COKE_TARGET_BASIS = "legacy_daily_charge_per_dpr_hot_metal"
# Keep the operational warning conservative. Performance statistics from the
# former charge-report/DPR target are not evidence for the new measured basis
# and must not be quoted as though they were.
STALE_AFTER_DAYS = 14
# A day whose residual sits this far from the window median is a data problem -
# a historian gap or a blowdown - not a bias to calibrate against.
OUTLIER_SIGMA = 3.0


@dataclass(frozen=True)
class CokeCalibration:
    """The offset, and everything needed to judge whether to trust it."""

    offset_kg_per_thm: float
    sample_days: int
    residual_sd_kg_per_thm: float
    window_days: int
    fitted_on: str = ""
    first_day: str = ""
    last_day: str = ""
    outliers_dropped: int = 0
    target_basis: str = ACTUAL_COKE_TARGET_BASIS
    notes: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """Enough days, and a spread that is not itself nonsense."""

        return (
            self.target_basis == ACTUAL_COKE_TARGET_BASIS
            and self.sample_days >= 20
            and self.residual_sd_kg_per_thm < 60.0
        )

    def age_days(self, today: date | None = None) -> int | None:
        """Days since the offset was fitted, or None if it was never fitted."""

        if not self.fitted_on:
            return None
        try:
            fitted = date.fromisoformat(self.fitted_on)
        except ValueError:
            return None
        return ((today or datetime.now(timezone.utc).date()) - fitted).days

    def is_stale(self, today: date | None = None) -> bool:
        age = self.age_days(today)
        return age is not None and age > STALE_AFTER_DAYS

    def apply(self, raw_coke_rate_kg_per_thm: float) -> float:
        """Corrected coke rate. Never returns a negative or absurd figure."""

        if not self.is_usable:
            return float(raw_coke_rate_kg_per_thm)
        return max(0.0, float(raw_coke_rate_kg_per_thm) - self.offset_kg_per_thm)


NO_CALIBRATION = CokeCalibration(
    offset_kg_per_thm=0.0, sample_days=0, residual_sd_kg_per_thm=0.0,
    window_days=DEFAULT_WINDOW_DAYS,
    notes=["no calibration on file for the measured plant coke-rate basis"],
)


def fit_offset(
    predicted: Sequence[float],
    actual: Sequence[float],
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    days: Sequence[Any] | None = None,
) -> CokeCalibration:
    """
    Mean over-prediction across the window, with outlier days dropped.

    Pure arithmetic on two sequences so it can be tested without a database.
    The caller decides what "recent" means and passes only those days.

    Args:
         - predicted: Sequence[float] - Energy-balance coke rate per day.
         - actual: Sequence[float] - Measured plant coke rate for the same days.
         - window_days: int - Recorded for provenance; the caller does the
           windowing.
         - days: Sequence[Any] | None - Day labels, used only for reporting.

    Returns:
         - return CokeCalibration - Offset to SUBTRACT from a raw prediction.
    """

    pairs = [
        (float(p), float(a))
        for p, a in zip(predicted, actual)
        if p is not None and a is not None
        and float(a) > 0.0 and float(p) == float(p) and float(a) == float(a)
    ]
    if not pairs:
        return NO_CALIBRATION

    residuals = [p - a for p, a in pairs]
    median = sorted(residuals)[len(residuals) // 2]
    spread = (sum((r - median) ** 2 for r in residuals) / len(residuals)) ** 0.5
    kept = [
        r for r in residuals
        if spread <= 0.0 or abs(r - median) <= OUTLIER_SIGMA * spread
    ]
    dropped = len(residuals) - len(kept)
    if not kept:
        return NO_CALIBRATION

    offset = sum(kept) / len(kept)
    sd = (sum((r - offset) ** 2 for r in kept) / len(kept)) ** 0.5
    notes: list[str] = []
    if len(kept) < 20:
        notes.append(f"only {len(kept)} usable days - offset is provisional")
    if sd > 25.0:
        notes.append(
            f"residual spread {sd:.0f} kg/tHM is wide; the offset corrects the "
            "average day but individual days will still scatter"
        )
    labels = [str(x) for x in (days or [])]
    return CokeCalibration(
        offset_kg_per_thm=offset,
        sample_days=len(kept),
        residual_sd_kg_per_thm=sd,
        window_days=window_days,
        fitted_on=datetime.now(timezone.utc).date().isoformat(),
        first_day=labels[0] if labels else "",
        last_day=labels[-1] if labels else "",
        outliers_dropped=dropped,
        notes=notes,
    )


def save_calibration(
    calibration: CokeCalibration, path: Path | str | None = None
) -> Path:
    """Persist atomically, so a crash mid-write cannot leave a half file."""

    target = Path(path) if path else CALIBRATION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "offset_kg_per_thm": calibration.offset_kg_per_thm,
        "sample_days": calibration.sample_days,
        "residual_sd_kg_per_thm": calibration.residual_sd_kg_per_thm,
        "window_days": calibration.window_days,
        "fitted_on": calibration.fitted_on,
        "first_day": calibration.first_day,
        "last_day": calibration.last_day,
        "outliers_dropped": calibration.outliers_dropped,
        "target_basis": calibration.target_basis,
        "notes": list(calibration.notes),
    }
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


def load_calibration(path: Path | str | None = None) -> CokeCalibration:
    """The stored offset, or a no-op that says so.

    A missing or unreadable file must never break a recommendation - it means
    nobody has run the refresh yet, and the raw figure is shown with a note.
    """

    target = Path(path) if path else CALIBRATION_PATH
    if not target.exists():
        return NO_CALIBRATION
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return NO_CALIBRATION
    try:
        target_basis = str(raw.get("target_basis", LEGACY_COKE_TARGET_BASIS))
        notes = list(raw.get("notes", []) or [])
        if target_basis != ACTUAL_COKE_TARGET_BASIS:
            notes.append(
                "stored offset used the previous coke-rate definition; refit "
                "against hourly COKE_CALC_MT per hot-metal tonne"
            )
        return CokeCalibration(
            offset_kg_per_thm=float(raw["offset_kg_per_thm"]),
            sample_days=int(raw.get("sample_days", 0)),
            residual_sd_kg_per_thm=float(raw.get("residual_sd_kg_per_thm", 0.0)),
            window_days=int(raw.get("window_days", DEFAULT_WINDOW_DAYS)),
            fitted_on=str(raw.get("fitted_on", "")),
            first_day=str(raw.get("first_day", "")),
            last_day=str(raw.get("last_day", "")),
            outliers_dropped=int(raw.get("outliers_dropped", 0)),
            target_basis=target_basis,
            notes=notes,
        )
    except (KeyError, TypeError, ValueError):
        return NO_CALIBRATION
