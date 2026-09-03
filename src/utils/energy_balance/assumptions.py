"""Every number in the energy balance that the plant has not measured.

The balance itself is an accounting identity, so its STRUCTURE needs no
calibration. But a handful of scalars inside it are not measured at this plant
and are currently carrying literature or assumed values. Those are collected
here, in one registry, so that:

  * an operator can replace a guess with a real figure the moment one exists,
    without touching yml or code;
  * nobody has to go hunting through four files to find out what is assumed;
  * every value states its BASIS and its CONFIDENCE, so a literature value is
    never mistaken for a measurement.

WHAT IS AND IS NOT IN HERE.

Physics is not negotiable and is not listed: 7.38 MJ/kg Fe for iron oxide
reduction, 32.8 MJ/kg for C to CO2, the CO and H2 calorific values, molar
volume. Those are constants of nature and an operator overriding them would be
breaking the balance, not calibrating it.

What IS listed is anything plant-specific that we could not measure and had to
supply: fuel hydrogen and carbon fractions, dust carbon, blast humidity, tap
temperature, and the enthalpies that depend on them.

CONFIDENCE LEVELS.

    measured    from a plant tag or lab report; shown for completeness only
    literature  a published value for this material's class or rank
    assumed     a working figure with no strong external support - these are
                the ones most worth replacing

Ordering within the table puts the highest-impact unknowns first, so the person
filling it in spends their effort where it changes the answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OVERRIDES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "energy_balance_assumptions.json"
)


@dataclass(frozen=True)
class Assumption:
    """One unmeasured number, with everything needed to show and edit it.

    Args:
         - key: str - Dotted path into the energy balance config, e.g.
           ``fuels.hydrogen_pct.pci``. This is what ``apply_overrides`` writes.
         - label: str - Operator-facing name.
         - unit: str - Unit shown beside the value.
         - default: float - What ships today.
         - basis: str - Where the default came from, in one line.
         - confidence: str - "measured", "literature" or "assumed".
         - impact: str - What changes if this is wrong. The reason an operator
           should care about filling it in.
         - minimum / maximum: float - Physically sensible bounds. Entering
           outside these is a data-entry slip, not a calibration.

    Returns:
         - return Assumption - One registry entry.
    """

    key: str
    label: str
    unit: str
    default: float
    basis: str
    confidence: str
    impact: str
    minimum: float
    maximum: float


# Ordered by how much a wrong value costs, worst first.
ASSUMPTIONS: tuple[Assumption, ...] = (
    Assumption(
        key="fuels.dust_carbon_pct.flue",
        label="Carbon in flue dust (dust catcher)",
        unit="%",
        default=22.3,
        basis="Plant XRF, BF-2, n=6: mean LoI 24.83% (range 15.2-44.3), taken "
              "at 90% to exclude the hydrogen and oxygen in volatile matter.",
        confidence="literature",
        impact="Carbon charged but never burnt. The 15-44% spread across spot "
               "samples is wide - a monthly composite would be worth more than "
               "another spot sample.",
        minimum=0.0,
        maximum=60.0,
    ),
    Assumption(
        key="fuels.dust_carbon_pct.gcp",
        label="Carbon in GCP dust",
        unit="%",
        default=47.1,
        basis="Plant XRF, BF-2, n=6: mean LoI 52.29% (range 45.2-55.4), taken "
              "at 90%. More than double the 20% previously assumed.",
        confidence="literature",
        impact="GCP dust is the fine carry-over fraction, so it is the "
               "CARBON-RICH stream, not the lean one - its Fe2O3 is 27% against "
               "the dust catcher's 55%. The old assumption had this backwards.",
        minimum=0.0,
        maximum=60.0,
    ),
    Assumption(
        key="fuels.carbon_fraction.pci",
        label="Carbon fraction of PCI",
        unit="fraction",
        default=0.75,
        basis="Currently 0.75. This coal's rank (22.4% VM daf, 9.2% ash) "
              "implies nearer 0.79 - see energy_balance.yml.",
        confidence="assumed",
        impact="Directly scales the largest input term. NOTE: raising it "
               "WIDENS the top-gas volume gap rather than closing it.",
        minimum=0.60,
        maximum=0.90,
    ),
    Assumption(
        key="fuels.carbon_fraction.coke",
        label="Carbon fraction of coke",
        unit="fraction",
        default=0.87,
        basis="Matches measured fixed carbon of 87.4%. Consistent, but on a "
              "dry basis - coke is charged with 2-4% total moisture.",
        confidence="literature",
        impact="Largest single input term. A 1% error is ~150 MJ/tHM.",
        minimum=0.75,
        maximum=0.95,
    ),
    Assumption(
        key="fuels.hydrogen_pct.pci",
        label="Hydrogen in PCI",
        unit="%",
        default=4.2,
        basis="Medium-volatile bituminous rank (4.5-4.9% daf) converted to "
              "the as-charged basis. No ultimate analysis exists or is coming.",
        confidence="literature",
        impact="Currently INERT: the fuel hydrogen term is switched off after "
               "being ruled out on evidence. Only matters if it is re-enabled.",
        minimum=2.0,
        maximum=7.0,
    ),
    Assumption(
        key="fuels.hydrogen_pct.coke",
        label="Hydrogen in coke",
        unit="%",
        default=0.35,
        basis="Carbonisation drives off nearly all hydrogen; BF coke runs "
              "0.3-0.5% and this coke's VM of 0.94% is at the low end.",
        confidence="literature",
        impact="Small either way, ~1 kg H/tHM. Also inert while the hydrogen "
               "term is off.",
        minimum=0.0,
        maximum=1.5,
    ),
    Assumption(
        key="blast_moisture_g_per_nm3",
        label="Moisture in the blast",
        unit="g/Nm³",
        default=15.0,
        basis="Plant-stated constant. There is no humidity tag in "
              "process_params, so this cannot vary with the weather as it "
              "really does.",
        confidence="assumed",
        impact="Arrives already oxidised so earns no input heat, but it "
               "gasifies carbon and shows up in top-gas H2. Monsoon and "
               "winter values differ materially.",
        minimum=0.0,
        maximum=40.0,
    ),
    Assumption(
        key="demand.hot_metal_mj_per_t",
        label="Hot metal enthalpy at tap",
        unit="MJ/t",
        default=1378.0,
        basis="Plant-stated tap temperature of 1500 °C.",
        confidence="measured",
        impact="Roughly 8% of the output side. Scales directly with tap "
               "temperature, so revise if the tapping practice changes.",
        minimum=1100.0,
        maximum=1600.0,
    ),
    Assumption(
        key="demand.slag_mj_per_kg",
        label="Slag enthalpy",
        unit="MJ/kg",
        default=1.80,
        basis="Standard figure for BF slag at tap temperature.",
        confidence="literature",
        impact="At ~330 kg slag/tHM this is ~600 MJ/tHM, so a 10% error is "
               "60 MJ/tHM.",
        minimum=1.2,
        maximum=2.4,
    ),
    Assumption(
        key="demand.burden_moisture_mj_per_kg",
        label="Burden moisture, heat to evaporate",
        unit="MJ/kg H₂O",
        default=2.70,
        basis="Latent heat plus sensible heat to top gas temperature.",
        confidence="literature",
        impact="Small on a dry day, larger in the monsoon when burden "
               "moisture climbs.",
        minimum=2.3,
        maximum=3.5,
    ),
)

BY_KEY: dict[str, Assumption] = {a.key: a for a in ASSUMPTIONS}


def load_overrides(path: Path | str | None = None) -> dict[str, float]:
    """Operator-supplied values, or an empty dict if none have been saved.

    A missing or unreadable file is not an error: it simply means nobody has
    filled the table in yet, and the shipped defaults apply.

    Args:
         - path: Path | str | None - Override file. Defaults to OVERRIDES_PATH.

    Returns:
         - return dict[str, float] - Mapping of assumption key to value.
    """

    target = Path(path) if path else OVERRIDES_PATH
    if not target.exists():
        return {}
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, float] = {}
    for key, value in (raw or {}).items():
        if key not in BY_KEY:
            continue  # a key retired since the file was written
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def save_overrides(
    overrides: dict[str, float], path: Path | str | None = None
) -> Path:
    """Persist operator values, dropping any that merely restate the default.

    Storing a value identical to the default would freeze it: a later change to
    the shipped figure would be silently overridden by a stale file that the
    operator never meant as an instruction.

    Args:
         - overrides: dict[str, float] - Key to value.
         - path: Path | str | None - Target file.

    Returns:
         - return Path - Where it was written.
    """

    target = Path(path) if path else OVERRIDES_PATH
    keep = {
        key: float(value)
        for key, value in overrides.items()
        if key in BY_KEY and not _is_default(key, value)
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(keep, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)
    return target


def _is_default(key: str, value: Any) -> bool:
    try:
        return abs(float(value) - BY_KEY[key].default) < 1e-12
    except (TypeError, ValueError, KeyError):
        return False


def clamp(key: str, value: float) -> float:
    """Hold a value inside its physically sensible band."""

    spec = BY_KEY[key]
    return max(spec.minimum, min(spec.maximum, float(value)))


def apply_overrides(
    cfg: dict[str, Any], overrides: dict[str, float] | None = None
) -> dict[str, Any]:
    """Return a copy of the config with operator values written in.

    The input config is not mutated - callers hold a cached singleton from
    ``load_config`` and a mutation there would leak into every later balance in
    the process.

    Args:
         - cfg: dict[str, Any] - Loaded energy balance config.
         - overrides: dict[str, float] | None - Values; loaded from disk if None.

    Returns:
         - return dict[str, Any] - Config with overrides applied.
    """

    values = load_overrides() if overrides is None else overrides
    if not values:
        return cfg
    out = _deep_copy(cfg)
    for key, value in values.items():
        if key not in BY_KEY:
            continue
        node: Any = out
        parts = key.split(".")
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = clamp(key, value)
    return out


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def current_values(overrides: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """The registry as rows ready for a table, defaults merged with overrides.

    Args:
         - overrides: dict[str, float] | None - Values; loaded from disk if None.

    Returns:
         - return list[dict[str, Any]] - One row per assumption, in registry order.
    """

    values = load_overrides() if overrides is None else overrides
    rows = []
    for spec in ASSUMPTIONS:
        supplied = values.get(spec.key)
        rows.append(
            {
                "key": spec.key,
                "Parameter": spec.label,
                "Value": float(supplied if supplied is not None else spec.default),
                "Unit": spec.unit,
                "Source": "operator" if supplied is not None else spec.confidence,
                "Default": spec.default,
                "Basis": spec.basis,
                "Why it matters": spec.impact,
            }
        )
    return rows
