"""What the Burden & Fuel Optimiser page shows beside every result (SV-3.3, #300): the plant's slag bands, the
register's plausibility bands, the facts the engine rests on, Fixed-ω and STRATA-ω side by side, and the assumptions.

The same sources as page 11: ``utils.slag_bands.PLANT_SLAG_BANDS`` (plant p5-p95 of B2, T-basicity and IB4; page 9
and page 11 caption every ratio with it), the fact register in force (``utils.burden_engine.facts.load_register``,
with an approved overlay of the fact register editor): ``FactRegister.band`` for BAND-01..03, ``table(ENGINE_FACTS)``
and ``model_table()`` (page 11's "Facts this result rests on"), and ``assets/burden_engine/assumptions.md`` (page 11's
"Assumptions and what the engine does not see"). Cheap: no model run, no database.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from utils.burden_engine.commentary import load_assumptions
from utils.burden_engine.facts import ENGINE_FACTS, load_register
from utils.slag_bands import PLANT_SLAG_BANDS

#: Labels and the role of each plant slag band on page 11 (``sections.render_plan_summary``).
SLAG_BAND_LABELS: dict[str, tuple[str, str, str]] = {
    "b2": ("B2", "CaO/SiO₂", "The ratio this page constrains (CaO/SiO₂ limits of the case)."),
    "t_basicity": ("T-basicity", "(CaO + MgO)/SiO₂", "Display only here (page 9 constrains it)."),
    "ib4": (
        "IB4",
        "(CaO + MgO)/(SiO₂ + Al₂O₃)",
        "Display only; reads lower than T-basicity because Al₂O₃ is in the denominator.",
    ),  # fmt: skip
}
#: Register bands the page draws: fuel rate (BAND-01), hot-metal temperature (BAND-02) and Si (BAND-03).
REGISTER_BANDS: dict[str, tuple[str, str, str]] = {
    "fuel_rate": ("BAND-01", "Fuel rate (coke wet + nut coke + PCI)", "kg/tHM"),
    "hmt": ("BAND-02", "Hot-metal temperature", "°C"),
    "hm_si": ("BAND-03", "Hot-metal Si", "%"),
}


class SlagBand(BaseModel):
    key: str = Field(..., description="b2, t_basicity or ib4 (the slag fields cao_sio2, t_basicity, ib4)")
    label: str
    ratio: str = Field(..., description="The ratio, e.g. CaO/SiO₂")
    lo: float = Field(..., description="Plant p5 over 6,515 hours of HM_SLAG analysis, dimensionless")
    hi: float = Field(..., description="Plant p95 over 6,515 hours of HM_SLAG analysis, dimensionless")
    note: str = Field(..., description="What the page does with the ratio")


class RegisterBand(BaseModel):
    key: str = Field(..., description="fuel_rate, hmt or hm_si")
    fact_id: str = Field(..., description="Fact of the band (BAND-01..03)")
    label: str
    unit: str
    lo: float = Field(..., description="Lower edge, in the band's unit")
    hi: float = Field(..., description="Upper edge, in the band's unit")


class FactRow(BaseModel):
    id: str
    statement: str
    value: float | str | None = Field(None, description="Value of the fact, in the fact's unit")
    unit: str | None = None
    uncertainty: str = Field("", description="± standard error or the range, in the fact's unit")
    status: str = Field(..., description="Model, Supported, Validated, Assumed, Modelled only, ...")
    grade: str | None = None
    next_test: str | None = None
    version: str | None = None


class ModelFactRow(BaseModel):
    id: str
    statement: str
    unit: str | None = None
    fixed_omega: float | None = Field(None, description="Fixed-ω response or realised value, in the fact's unit")
    strata_omega: float | None = Field(None, description="STRATA-ω response, in the fact's unit")
    sigma_strata: float | None = Field(None, description="1 sigma of the STRATA-ω response, in the fact's unit")
    status: str
    realised: float | None = Field(None, description="VAL: realised burnt coke, kg/tHM")
    n: int | None = Field(None, description="VAL: days tested, count")
    rmse_fixed: float | None = Field(None, description="VAL: out-of-sample RMSE of Fixed-ω, kg/tHM")
    rmse_strata: float | None = Field(None, description="VAL: out-of-sample RMSE of STRATA-ω, kg/tHM")
    skill: float | None = Field(None, description="VAL: skill of STRATA-ω over Fixed-ω, dimensionless")


class OptimiserReference(BaseModel):
    facts_version: str = Field(..., description="Fact register version in force")
    facts_date: str | None = Field(None, description="Date of the register")
    slag_bands: list[SlagBand] = Field(..., description="Plant slag basicity bands (utils.slag_bands)")
    bands: list[RegisterBand] = Field(..., description="Plausibility bands of the register the engine reports in")
    facts: list[FactRow] = Field(..., description="Facts the engine rests on (ENGINE_FACTS)")
    model_facts: list[ModelFactRow] = Field(..., description="MOD and VAL facts: Fixed-ω and STRATA-ω side by side")
    assumptions: str = Field(..., description="Assumptions and what the engine does not see (Markdown list)")


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def reference_view() -> OptimiserReference:
    reg = load_register()
    slag = [
        SlagBand(
            key=key,
            label=SLAG_BAND_LABELS[key][0],
            ratio=SLAG_BAND_LABELS[key][1],
            lo=lo,
            hi=hi,
            note=SLAG_BAND_LABELS[key][2],
        )  # fmt: skip
        for key, (lo, hi) in PLANT_SLAG_BANDS.items()
    ]
    bands = []
    for key, (fact_id, label, unit) in REGISTER_BANDS.items():
        lo, hi = reg.band(fact_id)
        bands.append(RegisterBand(key=key, fact_id=fact_id, label=label, unit=unit, lo=lo, hi=hi))
    facts = [FactRow.model_validate({k: (str(v) if k == "version" and v is not None else v) for k, v in row.items()})
             for row in reg.table(list(ENGINE_FACTS))]  # fmt: skip
    model = []
    for row in reg.model_table():
        n = row.get("n")
        model.append(
            ModelFactRow(
                id=row["id"],
                statement=row["statement"],
                unit=row.get("unit"),
                fixed_omega=_float(row.get("fixed_omega")),
                strata_omega=_float(row.get("strata_omega")),
                sigma_strata=_float(row.get("sigma_strata")),
                status=row["status"],
                realised=_float(row.get("realised")),
                n=int(n) if isinstance(n, (int, float)) else None,
                rmse_fixed=_float(row.get("rmse_fixed")),
                rmse_strata=_float(row.get("rmse_strata")),
                skill=_float(row.get("skill")),
            )
        )
    return OptimiserReference(
        facts_version=str(reg.version),
        facts_date=str(reg.date) if getattr(reg, "date", None) else None,
        slag_bands=slag,
        bands=bands,
        facts=facts,
        model_facts=model,
        assumptions=load_assumptions(),
    )
