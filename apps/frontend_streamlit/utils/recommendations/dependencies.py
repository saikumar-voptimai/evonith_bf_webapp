"""Derived/dependent parameter specifications for V-OptimAIse.

Defines :class:`ParamSpec` (a frozen dataclass) and :func:`build_bf_dependency_graph`
which returns the ordered list of blast furnace dependent parameters that the
optimiser appends to its output after the core recommendation is made.

Each :class:`ParamSpec` specifies the required input columns, a *formula*
(pure function) or a pre-trained sklearn *model*, and optional clipping bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd


class Regressor(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


Values = Dict[str, float]


@dataclass(frozen=True)
class ParamSpec:
    """Specification for a derived/dependent parameter.

    Exactly one of `formula` or `model` must be provided.

    - `inputs`: required column names.
    - `formula(values)`: returns the derived value.
    - `model`: sklearn-like regressor; inputs are taken in the order of `inputs`.
    - `clip`: optional (min,max) clamp for demo-safe guardbands.
    """

    name: str
    inputs: Tuple[str, ...]
    formula: Optional[Callable[[Values], float]] = None
    model: Optional[Regressor] = None
    clip: Optional[Tuple[float, float]] = None

    def compute(self, values: Values) -> float:
        missing = [k for k in self.inputs if k not in values or pd.isna(values[k])]
        if missing:
            raise KeyError(f"Missing inputs for '{self.name}': {missing}")

        has_formula = self.formula is not None
        has_model = self.model is not None
        if has_formula == has_model:
            raise ValueError(
                f"ParamSpec '{self.name}' must have exactly one of formula or model."
            )

        if self.formula is not None:
            y = float(self.formula(values))
        else:
            x = np.array([[float(values[k]) for k in self.inputs]], dtype=float)
            y = float(self.model.predict(x)[0])

        if self.clip is not None:
            lo, hi = self.clip
            y = float(np.clip(y, lo, hi))
        return y


class DependencyGraph:
    """Computes dependent parameters from a base row (true knobs + context).

    The graph resolves ordering automatically based on declared inputs.
    """

    def __init__(self, specs: Iterable[ParamSpec]) -> None:
        self._specs: Dict[str, ParamSpec] = {s.name: s for s in specs}
        self._order: List[str] = self._topological_order()

    def _topological_order(self) -> List[str]:
        names = set(self._specs.keys())

        deps: Dict[str, set[str]] = {}
        rev: Dict[str, set[str]] = {}

        for name, spec in self._specs.items():
            d = set([i for i in spec.inputs if i in names])
            deps[name] = d
            for i in d:
                rev.setdefault(i, set()).add(name)

        ready = [n for n in names if not deps[n]]
        order: List[str] = []

        while ready:
            n = ready.pop()
            order.append(n)
            for m in rev.get(n, set()):
                deps[m].discard(n)
                if not deps[m]:
                    ready.append(m)

        if len(order) != len(names):
            remaining = [n for n in names if deps[n]]
            raise ValueError(f"Dependency cycle detected among: {remaining}")

        return order

    def apply(self, row: Union[pd.Series, Dict[str, float]]) -> Dict[str, float]:
        values: Dict[str, float]
        if isinstance(row, pd.Series):
            values = {
                k: float(v) if v is not None and not pd.isna(v) else v
                for k, v in row.to_dict().items()
            }
        else:
            values = dict(row)

        for name in self._order:
            values[name] = self._specs[name].compute(values)

        return values

    def names(self) -> List[str]:
        return list(self._order)


@dataclass(frozen=True)
class BFColumns:
    """Column names used by dependency formulas.

    Adjust here if your dataset uses different labels.
    """

    wind_vol_nm3_hr: str = "HOT BLAST VOLUMENM3/HR."
    o2_enrich_pct: str = "O2 ENRICHMENT %"
    oxygen_flow_nm3_hr: str = "OXYGENFLOWNM3/HR."
    steam_kg_hr: str = "STEAMKGS/HR."

    hb_pressure_bar: str = "HOT BLAST PRESSUREBAR"
    top_pressure_bar: str = "TOPPRESSUREBAR"

    permeability: str = "PERMEABILITYKGS/HR."

    # internal derived names
    bosh_vol: str = "BOSH_VOL"
    k_value: str = "K_VALUE"


def _effective_o2_percent_from_flows(
    cbv: float, oxygen_flow: float, wind: float
) -> float:
    """Effective O2% in total wind (hot blast) given air (cbv) and pure O2 flow.

    Derived from your shared expression:
      (((CBV-oxygen_flow)*0.208 + oxygen_flow) / wind) * 100

    Notes:
    - Assumes CBV includes the oxygen_flow portion; i.e., air portion is (CBV - oxygen_flow).
    - Uses 20.8% baseline O2 in air.
    """
    if wind <= 0:
        raise ValueError("wind volume must be > 0")
    return (((cbv - oxygen_flow) * 0.208 + oxygen_flow) / wind) * 100.0


def oxygen_flow_from_enrichment(values: Values, cols: BFColumns) -> float:
    """Solve oxygen flow (Nm3/h) from enrichment % and volumes.

    You provided the enrichment coupling in the BoshVol expression.

    Let effectiveO2% = 20.8 + enrichment.
    effectiveO2%/100 = (0.208*(CBV - OF) + OF) / wind
    => OF = (wind*(effectiveO2%/100) - 0.208*CBV) / 0.792

    We default CBV to wind volume if CBV isn't separately available.
    """
    wind = float(values[cols.wind_vol_nm3_hr])
    enr = float(values[cols.o2_enrich_pct])

    cbv = float(
        values.get("CBV", wind)
    )  # optional override if you later add CBV column
    effective = 20.8 + enr
    of = (wind * (effective / 100.0) - 0.208 * cbv) / 0.792
    return float(max(of, 0.0))


def bosh_vol_from_formula(values: Values, cols: BFColumns) -> float:
    """Compute BoshVol from your provided formula.

    K_Value formula uses:
      BoshVol = (COVol + H2Vol + N2Vol) / 60

    where:
      COVol + H2Vol + N2Vol =
        2*(CBV*(20.8 + (effectiveO2% - 20.8))/100)
        + (20*(wind - oxygen_flow))/18000
        + (steam*1000)/18000

    Note: (20.8 + (effectiveO2% - 20.8)) == effectiveO2%.
    """
    wind = float(values[cols.wind_vol_nm3_hr])
    oxygen_flow = float(values[cols.oxygen_flow_nm3_hr])
    steam = float(values[cols.steam_kg_hr])

    cbv = float(values.get("CBV", wind))

    effective_o2_pct = _effective_o2_percent_from_flows(
        cbv=cbv, oxygen_flow=oxygen_flow, wind=wind
    )

    term1 = 2.0 * (cbv * (effective_o2_pct / 100.0))
    term2 = (20.0 * (wind - oxygen_flow)) / 18000.0
    term3 = (steam * 1000.0) / 18000.0

    co_h2_n2 = term1 + term2 + term3
    bosh_vol = co_h2_n2 / 60.0
    return float(bosh_vol)


def k_value_from_formula(values: Values, cols: BFColumns) -> float:
    """Compute K_Value (inverse permeability proxy) from your formula."""
    hb_p = float(values[cols.hb_pressure_bar])
    top_p = float(values[cols.top_pressure_bar])
    bosh_vol = float(values[cols.bosh_vol])

    if bosh_vol <= 0:
        raise ValueError("BoshVol must be > 0")

    # ((HB*1000+1033)^2 - (Top*1000+1033)^2)/(BoshVol^1.7)
    num = (hb_p * 1000.0 + 1033.0) ** 2 - (top_p * 1000.0 + 1033.0) ** 2
    den = bosh_vol**1.7
    return float(num / den)


def permeability_from_k_value(
    values: Values, cols: BFColumns, eps: float = 1e-9
) -> float:
    """Convert inverse-permeability proxy to permeability.

    You said: inverse(permeability) is calculated as per the shared formula.
    This returns a simple reciprocal.

    If later you want a calibrated mapping, replace this with: a / (k + b).
    """
    k = float(values[cols.k_value])
    return float(1.0 / max(k, eps))


def build_bf_dependency_graph(cols: BFColumns | None = None) -> DependencyGraph:
    """Factory for a demo-ready dependency graph using your provided formulas."""
    cols = cols or BFColumns()

    specs = [
        ParamSpec(
            name=cols.oxygen_flow_nm3_hr,
            inputs=(cols.wind_vol_nm3_hr, cols.o2_enrich_pct),
            formula=lambda v: oxygen_flow_from_enrichment(v, cols),
            clip=(0.0, 1e7),
        ),
        ParamSpec(
            name=cols.bosh_vol,
            inputs=(
                cols.wind_vol_nm3_hr,
                cols.oxygen_flow_nm3_hr,
                cols.steam_kg_hr,
            ),
            formula=lambda v: bosh_vol_from_formula(v, cols),
            clip=(0.0, 1e9),
        ),
        ParamSpec(
            name=cols.k_value,
            inputs=(cols.hb_pressure_bar, cols.top_pressure_bar, cols.bosh_vol),
            formula=lambda v: k_value_from_formula(v, cols),
        ),
        ParamSpec(
            name=cols.permeability,
            inputs=(cols.k_value,),
            formula=lambda v: permeability_from_k_value(v, cols),
            clip=(0.0, 1e9),
        ),
    ]

    return DependencyGraph(specs)
