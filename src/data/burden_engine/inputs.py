"""Yard, prices and limits for the burden engine, from what the Blend Mix Optimiser already maintains (#253).

Both pages see the same yard: ore chemistry and stock come from ``EvonithBmoContextProvider`` (Neon lab tables and
stock), prices / share bounds / selection / flux limits / fuel prices / slag limits from the operator's saved inputs
(``services.operator_inputs``: the version in force in ``twin.operator_inputs``, ``bmo_operator_inputs.yml`` as seed
and fallback; SV-3.1). The pure functions turn those typed records into the engine's input dict; only
:func:`load_yard` does I/O.

Engine material record: id, name, code, kind (sinter / ore / pellet), price (Rs/wet t), lo / hi (wet % of ferrous),
selected, chem (dry %: Fe, FeO, SiO2, Al2O3, CaO, MgO, MnO, TiO2, P, LOI), moist (%), max_t_day (stock / horizon),
k_rule (burden-resistance rule tag) plus provenance (chem_source, chem_time, carried, reason).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import pandas as pd

if TYPE_CHECKING:  # the BMO package init is heavy; the records are only read by attribute here
    from utils.bmo.types import FluxInput, OreChemistry, OreInput

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
FE_IN_FE2O3 = 2 * 55.845 / 159.69


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def kind_for(material_key: str) -> str:
    key = (material_key or "").lower()
    if key.startswith("sinter"):
        return "sinter"
    if key.startswith("pellet"):
        return "pellet"
    return "ore"


def chem_from_ore(c: OreChemistry) -> tuple[dict[str, float], float]:
    """Engine chemistry (dry %) and moisture from a BMO chemistry record."""
    chem = dict(Fe=c.fe_t_pct, FeO=c.feo_pct, SiO2=c.sio2_pct, Al2O3=c.al2o3_pct, CaO=c.cao_pct, MgO=c.mgo_pct,
                MnO=c.mno_pct, TiO2=c.tio2_pct, P=c.p_pct, LOI=getattr(c, "loi_pct", 0.0))
    return {k: float(v or 0.0) for k, v in chem.items()}, float(c.moisture_pct or 0.0)


def chem_from_flux(f: FluxInput) -> dict[str, float]:
    return dict(Fe=float(f.fe2o3_pct or 0.0) * FE_IN_FE2O3, FeO=0.0, SiO2=float(f.sio2_pct), Al2O3=float(f.al2o3_pct),
                CaO=float(f.cao_pct), MgO=float(f.mgo_pct), MnO=float(f.mno_pct), TiO2=float(f.tio2_pct),
                P=float(f.p_pct), LOI=float(f.loi_pct))


def _saved(prefs: Mapping[str, Any] | None, *keys: str) -> dict:
    d: Any = prefs or {}
    for k in keys:
        d = (d or {}).get(k, {}) if isinstance(d, Mapping) else {}
    return dict(d or {})


def yard_rows(ores: Iterable[OreInput], prefs: Mapping[str, Any] | None = None, *,
              exclude_fallback: bool = True, k_rules: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """One engine material record per configured ore, with the operator's saved price / shares / selection on top.
    Materials whose chemistry fell back to configured defaults, or with no Fe, are marked ``carried=False``."""
    saved = _saved(prefs, "ore_editor", "rows")
    selected_ids = set((_saved(prefs, "ore_editor").get("selected_ore_ids") or []))
    rows = []
    for o in ores:
        chem, moist = chem_from_ore(o.chemistry)
        s = saved.get(o.ore_id, {}) or {}
        source = str(o.metadata.get("chemistry_source", "") or "")
        carried, reason = True, ""
        if chem["Fe"] <= 0:
            carried, reason = False, "no Fe in the chemistry"
        elif exclude_fallback and source == "fallback":
            carried, reason = False, "no lab chemistry in the window (configured fallback values only)"
        code = str(o.metadata.get("material_key", "") or "")
        rows.append(dict(
            id=o.ore_id, name=o.display_name, code=code, kind=kind_for(code),
            price=float(s.get("price_rs_per_mt", o.price_rs_per_mt)),
            lo=float(s.get("min_share_pct", o.min_share_pct)), hi=float(s.get("max_share_pct", o.max_share_pct)),
            selected=(o.ore_id in selected_ids) if selected_ids else True,
            stock_mt=float(o.stock_mt), chem=chem, moist=moist, k_rule=(k_rules or {}).get(o.ore_id),
            chem_source=source or "lab", chem_time=str(o.metadata.get("chemistry_sample_timestamp", "") or ""),
            chem_rows=int(o.metadata.get("chemistry_rows_used", 0) or 0), carried=carried, reason=reason,
        ))
    return rows


MN_IN_MNO = 54.938 / 70.937
TI_IN_TIO2 = 0.599341397


def catalogue_fluxes(flux_cfg: Iterable[Mapping[str, Any]]) -> list[Any]:
    """The flux catalogue page 9 offers its LP (``setting_bmo.yml`` ``bmo.flux_inputs``), as flux records.

    Same rules as the Blend Mix Optimiser's flux editor: hidden rows are skipped, Mn / Ti given on an element
    basis are converted to MnO / TiO2, only optimizable rows are carried. (The context provider's
    ``get_flux_inputs`` is the flux charged in the last charge, keyed by material code and without prices, so it
    is not a catalogue.)"""
    from types import SimpleNamespace

    out = []
    for item in flux_cfg or []:
        if bool(item.get("hidden_by_default", False)):
            continue
        g = lambda k: float(item.get(k, 0.0) or 0.0)  # noqa: E731
        mno = g("mn_pct") / MN_IN_MNO if str(item.get("mn_basis", "mno")).lower() == "mn" else g("mno_pct")
        tio2 = g("ti_pct") / TI_IN_TIO2 if str(item.get("ti_basis", "tio2")).lower() == "ti" else g("tio2_pct")
        fid = str(item.get("flux_id", ""))
        out.append(SimpleNamespace(
            flux_id=fid, display_name=str(item.get("display_name", fid)),
            enabled=bool(item.get("enabled", True)) and bool(item.get("optimizable", False)),
            price_rs_per_mt=g("price_rs_per_mt"), stock_mt=g("stock_mt"),
            max_qty_mt=(float(item["max_qty_mt"]) if item.get("max_qty_mt") is not None else None),
            moisture_pct=g("moisture_pct"), sio2_pct=g("sio2_pct"), al2o3_pct=g("al2o3_pct"), cao_pct=g("cao_pct"),
            mgo_pct=g("mgo_pct"), fe2o3_pct=g("fe2o3_pct"), mno_pct=mno, tio2_pct=tio2, p_pct=g("p_pct"),
            loi_pct=g("loi_pct")))
    return out


def flux_rows(fluxes: Iterable[FluxInput], prefs: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    saved = _saved(prefs, "flux_editor", "rows")
    rows = []
    for f in fluxes:
        s = saved.get(f.flux_id, {}) or {}
        chem = chem_from_flux(f)
        max_qty = s.get("max_qty_mt", f.max_qty_mt)
        rows.append(dict(
            id=f.flux_id, name=f.display_name, code=f.flux_id, kind="flux",
            price=float(s.get("price_rs_per_mt", f.price_rs_per_mt)),
            stock_mt=float(s.get("stock_mt", f.stock_mt)), max_qty_mt=(None if max_qty is None else float(max_qty)),
            chem=chem, moist=float(f.moisture_pct),
            loi_type="CO2" if chem["CaO"] + chem["MgO"] > 20.0 else "H2O", carried=bool(f.enabled),
        ))
    return rows


def fuel_prices(prefs: Mapping[str, Any] | None, default: Mapping[str, float] | None = None) -> dict[str, float]:
    rows = _saved(prefs, "fuel_ash_editor", "rows")
    d = dict(default or {})
    for key, name in (("coke_price", "coke"), ("nut_price", "nut_coke"), ("pci_price", "pci"), ("slag_price", "slag")):
        v = (rows.get(name) or {}).get("price_rs_per_mt")
        if v is not None:
            d[key] = float(v)
    return d


def slag_limits(prefs: Mapping[str, Any] | None) -> dict[str, float]:
    mi = _saved(prefs, "model_inputs")
    m = dict(B2_min="target_slag_basicity_min", B2_max="target_slag_basicity_max", slag_max="target_slag_rate_kg_per_thm",
             Al2O3_max="target_slag_al2o3_max_pct", MgO_min="target_slag_mgo_min_pct",
             MgO_Al2O3_min="target_slag_mgo_al2o3_ratio_min")
    return {k: float(mi[v]) for k, v in m.items() if mi.get(v) is not None}


def saved_charging(prefs: Mapping[str, Any] | None) -> dict[str, float]:
    mi = _saved(prefs, "model_inputs")
    out = {}
    if mi.get("charge_mass_mt") is not None:
        out["charge_t"] = float(mi["charge_mass_mt"])
    if mi.get("max_charges_per_hour") is not None:
        out["charges_per_h"] = float(mi["max_charges_per_hour"])
    return out


def build_engine_inputs(materials: Iterable[Mapping[str, Any]], fluxes: Iterable[Mapping[str, Any]],
                        fuel: Mapping[str, float], *, stock_horizon_days: float = 1.0,
                        use_stock: bool = True) -> dict[str, Any]:
    """The engine's input dict from (possibly operator-edited) yard and flux rows.

    Only carried and selected materials are offered to the LP. Stock caps a material's t/day at stock / horizon
    (``use_stock``); a material with zero stock is left out, as the Blend Mix Optimiser pins it to zero."""
    mats, left_out = [], {}
    for m in materials:
        if not m.get("carried", True):
            left_out[m["id"]] = m.get("reason", "not carried")
            continue
        rec = {k: m[k] for k in ("id", "name", "code", "kind", "price", "lo", "hi", "chem", "moist") if k in m}
        rec["selected"] = bool(m.get("selected", True))
        rec["k_rule"] = m.get("k_rule")
        if use_stock and m.get("stock_mt") is not None:
            if float(m["stock_mt"]) <= 0 and rec["selected"]:
                left_out[m["id"]] = "zero stock"
                rec["selected"] = False
            rec["max_t_day"] = max(0.0, float(m["stock_mt"])) / max(stock_horizon_days, 1e-9)
        mats.append(rec)
    fl = []
    for f in fluxes:
        if not f.get("carried", True):
            continue
        caps = [f.get("max_qty_mt")]
        if use_stock and f.get("stock_mt") is not None:
            caps.append(max(0.0, float(f["stock_mt"])) / max(stock_horizon_days, 1e-9))
        caps = [c for c in caps if c is not None]
        fl.append(dict(id=f["id"], name=f.get("name", f["id"]), code=f.get("code", f["id"]), kind="flux",
                       price=float(f["price"]), max_t_day=(min(caps) if caps else None), chem=f["chem"],
                       moist=float(f.get("moist", 0.0)), loi_type=f.get("loi_type", "CO2")))
    missing = {"coke_price", "nut_price", "pci_price"} - set(fuel)
    if missing:
        raise ValueError(f"fuel prices missing: {sorted(missing)}")
    if not any(m["selected"] for m in mats):
        raise ValueError("no ferrous material is selected, carried and in stock")
    return dict(source="Blend Mix Optimiser yard (context provider + operator inputs)", materials=mats,
                fluxes=fl, not_carried=left_out, fuel={k: float(v) for k, v in fuel.items()})


FERROUS_PREFIX = ("sinter_", "ore_", "pellet_")


def charged_blend(charges: pd.DataFrame, materials: Iterable[Mapping[str, Any]], *, hours: float = 24.0) -> dict[str, Any]:
    """Today's ferrous blend: wet shares of what was charged in the last ``hours`` before the latest charge.

    ``charges``: ``charge_data`` rows (DatetimeIndex, ``<code>_mt`` columns). Shares are of all ferrous charged; a
    ferrous code with no yard material (or none with lab chemistry) is reported under ``unmapped`` and left out of
    ``shares``, which are then the carried materials' shares of the ferrous total (they sum to 100 - unmapped %)."""
    out: dict[str, Any] = dict(shares={}, unmapped={}, charges=0, window=("", ""))
    if charges is None or charges.empty or not isinstance(charges.index, pd.DatetimeIndex):
        return out
    if charges.index.tz is None:                     # Neon timestamps are UTC
        charges = charges.tz_localize("UTC")
    end = charges.index.max()
    frame = charges[charges.index > end - pd.Timedelta(hours=float(hours))]
    totals = {c[:-3]: float(pd.to_numeric(frame[c], errors="coerce").fillna(0.0).sum())
              for c in frame.columns if c.endswith("_mt") and c.startswith(FERROUS_PREFIX)}
    ferrous = sum(totals.values())
    if ferrous <= 0:
        return out
    by_code = {str(m.get("code", "")): m for m in materials if m.get("code")}
    for code, t in totals.items():
        if t <= 0:
            continue
        m = by_code.get(code)
        pct = 100.0 * t / ferrous
        if m is not None and m.get("carried", True):
            out["shares"][m["id"]] = out["shares"].get(m["id"], 0.0) + pct
        else:
            out["unmapped"][(m or {}).get("name", code)] = pct
    out.update(charges=int(len(frame)), ferrous_t=ferrous,
               window=(str(frame.index.min().tz_convert("Asia/Kolkata").strftime("%d %b %H:%M")),
                       str(end.tz_convert("Asia/Kolkata").strftime("%d %b %H:%M"))))
    return out


def load_charged_blend(materials: Iterable[Mapping[str, Any]], *, hours: float = 24.0) -> dict[str, Any]:
    """:func:`charged_blend` on ``offline_feed.charge_data`` (Neon), for the materials' codes the table has."""
    from datetime import datetime, timedelta, timezone

    from furnace_data.offline import OFFLINE_TABLES, fetch_offline_data

    mats = list(materials)
    allowed = OFFLINE_TABLES.get("offline_feed.charge_data") or set()
    cols = sorted(c for c in allowed if c.endswith("_mt") and c.startswith(FERROUS_PREFIX))
    end = datetime.now(timezone.utc)
    df = fetch_offline_data("offline_feed.charge_data", (end - timedelta(days=3), end), query_type="raw",
                            columns=["date_time", *cols])
    return charged_blend(df, mats, hours=hours)


def load_yard(cfg: Mapping[str, Any], *, prefs: Mapping[str, Any] | None = None,
              bmo_settings_path: str | Path = "src/config/setting_bmo.yml", provider: Any = None) -> dict[str, Any]:
    """Live yard from the BMO context provider (Neon chemistry + stock) and the operator's saved inputs.

    Fluxes are page 9's catalogue (``setting_bmo.yml``) with the operator's saved prices and limits. The operator
    inputs are ``prefs`` when given, else the version in force from the provider (``get_operator_inputs``:
    ``twin.operator_inputs`` in Neon, the YAML file as fallback, SV-3.1), the source page 9 and the API read.

    Returns dict(materials=[...], fluxes=[...], fuel={...}, slag={...}, charging_saved={...}, warnings=[...],
    operator_inputs={source, version, valid_from, author, reason})."""
    import yaml

    yard_cfg = dict(cfg.get("yard", {}))
    if provider is None:
        from data.bmo.context_provider import EvonithBmoContextProvider
        provider = EvonithBmoContextProvider()
    provenance: dict[str, Any] = {"source": "given"}
    if prefs is None:
        reader = getattr(provider, "get_operator_inputs", None)
        if not callable(reader):  # a provider without the accessor (tests): the same service
            from services.operator_inputs import load_current as reader
        current = reader()
        prefs, provenance = current.inputs, current.provenance()
    mode = str(yard_cfg.get("chemistry_mode", "avg"))
    days = int(yard_cfg.get("chemistry_window_days", 30))
    ores, diag = provider.build_ore_inputs(mode=mode, window_days=days)
    bmo_cfg = (yaml.safe_load(repo_path(bmo_settings_path).read_text(encoding="utf-8")) or {}).get("bmo", {})
    fluxes = catalogue_fluxes(bmo_cfg.get("flux_inputs", []))
    warnings = list(diag.get("warnings", []) if isinstance(diag, Mapping) else [])
    if not any(f.enabled for f in fluxes):
        warnings.append("No optimizable flux in setting_bmo.yml flux_inputs.")
    reason = str(provenance.get("reason") or "")
    if provenance.get("source") == "yaml" and ("not reachable" in reason or "no operator_inputs table" in reason):
        warnings.append(f"Saved prices and limits from bmo_operator_inputs.yml: {reason}")
    return dict(
        materials=yard_rows(ores, prefs, exclude_fallback=bool(yard_cfg.get("exclude_fallback_chemistry", True)),
                            k_rules=yard_cfg.get("k_rules")),
        fluxes=flux_rows(fluxes, prefs),
        fuel=fuel_prices(prefs, dict(slag_price=float(((cfg.get("prices") or {}).get("slag_price_rs_per_t")) or 0.0))),
        slag=slag_limits(prefs),
        charging_saved=saved_charging(prefs),
        warnings=warnings,
        operator_inputs=provenance,
    )
