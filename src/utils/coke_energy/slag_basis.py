"""The Blend Mix Optimiser's slag balance on a two-zone case, per tonne of hot metal.

``utils.bmo.slag_balance.calculate_full_slag_balance`` (page 9) is the plant's component ledger: every oxide and
minor element of the ores, fluxes, fuel ash and top-gas dust, the pig iron closed on held hot-metal chemistry, then
SiO2 consumed by Si reduction, FeO from the Fe that does not reach the pig iron, MnO and TiO2 from what is not
reduced, sulphur less the hot metal's and a gas loss, alkalis by a slag fraction, all summed. This module is the
same sequence on the engine's case (burden in dry kg/tHM, fuels in dry kg/tHM, dust streams in kg/tHM), so that the
Burden & Fuel Optimiser reports the slag rate and basicity page 9 would for the same blend (user decision
11 Nov 2026: page 9's conventions are the plant's).

What the engine takes from page 9 here (``rules``):

* hot-metal Si, S, C held for the slag side (``slag_balance`` in ``setting_bmo.yml``, operator-editable on page 9);
  the thermal Si of the energy balance (``Settings.hm_Si``) is untouched;
* top-gas dust per stream (dust catcher, GCP) at the DPR's kg per charge with the measured chemistry, MgO and
  alkalis included;
* the minor oxides from the analyses (alkalis x ``alkali_to_slag_fraction``, sulphur less the pig iron's and
  ``sulphur_gas_loss_pct``) instead of a flat allowance;
* FeO from ``1 - fe_to_pig_iron_fraction`` of the Fe charged less the dust's, instead of a slag FeO %.

Without ``p['slag_basis']`` (or dust streams) ``bf_balance.slag_and_flux`` keeps its own rule, so the Coke Energy
pipeline, the twin and every calibration are unchanged. The parity with page 9 is tested in
``tests/test_burden_engine_slag_basis.py`` (same blend through both ledgers, 1e-9 relative).
"""

from __future__ import annotations

from typing import Any, Mapping

# the same constants as utils.bmo.types / utils.bmo.slag_balance (no import: this package stays Streamlit-free and
# light; the parity test holds them equal)
MN_IN_MNO = 0.774461846
TI_IN_TIO2 = 0.599341397
FE_IN_FE2O3 = 111.69 / 159.69

COMPONENTS = ("sio2", "al2o3", "cao", "mgo", "mn", "p", "s", "ti", "zn", "alkali", "fe", "caf2")

# fuels of the two-zone case -> page 9's fuel ids
FUEL_KEYS = (("coke", "coke"), ("nut", "nut_coke"), ("pci", "pci"))


def _f(v: Any) -> float:
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _pct(v: Any) -> float:
    """Page 9's ``_safe_pct``: a percentage clamped to 0-99."""
    return min(max(_f(v), 0.0), 99.0)


def _mass(dry_kg: float, pct: Any) -> float:
    return max(0.0, _f(dry_kg)) * (_pct(pct) / 100.0)


def _dry(wet_kg: float, moist_pct: Any) -> float:
    return max(0.0, _f(wet_kg)) * (100.0 - _pct(moist_pct)) / 100.0


def zero() -> dict[str, float]:
    return {k: 0.0 for k in COMPONENTS}


def add(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    return {k: _f(a.get(k)) + _f(b.get(k)) for k in COMPONENTS}


def subtract_clamped(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    """Page 9's dust deduction: never below zero per component."""
    return {k: max(0.0, _f(a.get(k)) - _f(b.get(k))) for k in COMPONENTS}


# ---------------------------------------------------------------------------------------------- components
def slag_forming_kg(comp: Mapping[str, float]) -> float:
    """Page 9's "slag-forming oxides" of one material (its ``Slag (MT)`` column, ``compute_slag_contribution_mt``):
    SiO2 + Al2O3 + CaO + MgO + TiO2 + MnO + Na2O + K2O (Mn and Ti back as their oxides); S, P, Zn, CaF2 and Fe left out."""
    return (_f(comp.get("sio2")) + _f(comp.get("al2o3")) + _f(comp.get("cao")) + _f(comp.get("mgo"))
            + _f(comp.get("mn")) / MN_IN_MNO + _f(comp.get("ti")) / TI_IN_TIO2 + _f(comp.get("alkali")))


def material_components(dry_kg: float, c: Mapping[str, Any]) -> dict[str, float]:
    """One ore, sinter, pellet or flux (dry kg; comp in dry %), as page 9's ore and flux ledgers.

    Fe is the total Fe of the material (``Fe``; a flux's Fe2O3 is already converted by the inputs adapter). MnO and
    TiO2 are carried as the elements, alkalis as Na2O + K2O, like page 9."""
    kg = max(0.0, _f(dry_kg))
    t = zero()
    t["fe"] = _mass(kg, c.get("Fe"))
    t["sio2"] = _mass(kg, c.get("SiO2"))
    t["al2o3"] = _mass(kg, c.get("Al2O3"))
    t["cao"] = _mass(kg, c.get("CaO"))
    t["mgo"] = _mass(kg, c.get("MgO"))
    t["mn"] = _mass(kg, c.get("MnO")) * MN_IN_MNO
    t["p"] = _mass(kg, c.get("P"))
    t["s"] = _mass(kg, c.get("S"))
    t["ti"] = _mass(kg, c.get("TiO2")) * TI_IN_TIO2
    t["zn"] = _mass(kg, c.get("Zn"))
    t["alkali"] = _mass(kg, _pct(c.get("Na2O")) + _pct(c.get("K2O")))
    t["caf2"] = _mass(kg, c.get("CaF2"))
    return t


def burden_kind(m: Mapping[str, Any]) -> str:
    """``flux`` or the ferrous kind of a burden row: the engine's ``kind`` (sinter / ore / pellet / flux), else the
    calibration's class name (sinter, pellet, ore, flux)."""
    return str(m.get("kind") or ("flux" if m.get("name") == "flux" else m.get("name", "ore")))


def burden_by_material(case: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each burden row of the case: kind, dry kg, its components (kg) and page 9's per-material Fe and slag."""
    out: dict[str, dict[str, Any]] = {}
    for m in case["burden"]:
        comp = material_components(m["dry_kg"], m["comp"])
        out[str(m["name"])] = dict(kind=burden_kind(m), dry_kg=max(0.0, _f(m["dry_kg"])), comp=comp, fe_kg=comp["fe"],
                                   slag_kg=slag_forming_kg(comp))
    return out


def burden_components(case: Mapping[str, Any], p: Mapping[str, Any], extra_lime: float = 0.0,
                      extra_qtz: float = 0.0) -> dict[str, float]:
    """Ores and fluxes of the case (dry kg/tHM; comp in dry %), as page 9's ore and flux ledgers
    (:func:`material_components` summed), plus the pipeline's basicity-correction limestone / quartzite."""
    t = zero()
    for m in case["burden"]:
        t = add(t, material_components(m["dry_kg"], m["comp"]))
    t = add(t, correction_flux_components(p, extra_lime, extra_qtz))
    return t


def correction_flux_components(p: Mapping[str, Any], extra_lime: float = 0.0, extra_qtz: float = 0.0) -> dict[str, float]:
    """Basicity-correction limestone / quartzite of the pipeline's solver (kg/tHM; none in the engine's plans)."""
    t = zero()
    t["cao"] += extra_lime * p["lime_CaO"] + extra_qtz * p["qtz_CaO"]
    t["mgo"] += extra_lime * p["lime_MgO"]
    t["sio2"] += extra_lime * p["lime_SiO2"] + extra_qtz * p["qtz_SiO2"]
    t["al2o3"] += extra_qtz * p["qtz_Al2O3"]
    return t


def fuel_by_fuel(K: float, case: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Coke (dry K), nut coke and PCI as page 9's fuel-ash ledger: oxides on the ash, alkalis on the ash
    (``ash_comp['alk']``, else Na2O + K2O), sulphur and phosphorus on the dry fuel (``S_pct``, ``P_pct``); each with
    its dry kg, ash kg, components and page 9's per-fuel slag (the ash's slag-forming oxides), and the analysis it
    rests on (``moist_pct``, ``ash_pct``, ``ash_comp`` in % of the ash, ``S_pct`` / ``P_pct`` of the dry fuel; page 9's
    fuel-ash inputs) with the fuel as charged (``wet_kg``)."""
    out: dict[str, dict[str, Any]] = {}
    for name, kg in (("coke", K), ("nut", case["nut_kg"]), ("pci", case["pci_kg"])):
        t = zero()
        ash = 0.0
        fu = case.get(name) or {}
        a = fu.get("ash_comp") or {}
        alk = _pct(a.get("alk"))
        if alk <= 0.0:
            alk = _pct(a.get("Na2O")) + _pct(a.get("K2O"))
        if kg > 0:
            ash = _f(kg) * _pct(fu["ash"]) / 100.0
            t["sio2"] = _mass(ash, a.get("SiO2"))
            t["al2o3"] = _mass(ash, a.get("Al2O3"))
            t["cao"] = _mass(ash, a.get("CaO"))
            t["mgo"] = _mass(ash, a.get("MgO"))
            t["fe"] = _mass(ash, a.get("Fe2O3")) * FE_IN_FE2O3
            t["mn"] = _mass(ash, a.get("MnO")) * MN_IN_MNO
            t["ti"] = _mass(ash, a.get("TiO2")) * TI_IN_TIO2
            t["alkali"] = _mass(ash, alk)
            t["s"] = _mass(kg, fu.get("S_pct"))
            t["p"] = _mass(kg, fu.get("P_pct"))
        moist = _pct(fu.get("moist_pct"))
        dry = max(0.0, _f(kg))
        out[name] = dict(dry_kg=dry, wet_kg=dry / (1 - moist / 100.0) if moist < 100.0 else dry, ash_kg=ash, comp=t,
                         slag_kg=slag_forming_kg(t), moist_pct=moist, ash_pct=_pct(fu.get("ash")),
                         ash_comp={k: _pct(a.get(k)) for k in ("SiO2", "Al2O3", "CaO", "MgO", "Fe2O3", "MnO", "TiO2")}
                         | {"alk": alk},
                         S_pct=_pct(fu.get("S_pct")), P_pct=_pct(fu.get("P_pct")))
    return out


def fuel_components(K: float, case: Mapping[str, Any]) -> dict[str, float]:
    """The three fuels' components summed (:func:`fuel_by_fuel`)."""
    t = zero()
    for f in fuel_by_fuel(K, case).values():
        t = add(t, f["comp"])
    return t


def dust_by_stream(du: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each enabled dust stream (dry basis), as page 9 deducts it: kg (wet), dry kg, components, Fe and the
    slag-forming oxides it carries away."""
    out: dict[str, dict[str, Any]] = {}
    for d in du.get("streams") or []:
        if not d.get("enabled", True):
            continue
        kg = _dry(d.get("kg", 0.0), d.get("moist_pct", 0.0))
        t = zero()
        t["sio2"] = _mass(kg, d.get("SiO2_pct"))
        t["al2o3"] = _mass(kg, d.get("Al2O3_pct"))
        t["cao"] = _mass(kg, d.get("CaO_pct"))
        t["mgo"] = _mass(kg, d.get("MgO_pct"))
        t["fe"] = _mass(kg, d.get("Fe_pct"))
        t["mn"] = _mass(kg, d.get("Mn_pct"))
        t["p"] = _mass(kg, d.get("P_pct"))
        t["s"] = _mass(kg, d.get("S_pct"))
        t["ti"] = _mass(kg, d.get("Ti_pct"))
        t["zn"] = _mass(kg, d.get("Zn_pct"))
        t["alkali"] = _mass(kg, _pct(d.get("Na2O_pct")) + _pct(d.get("K2O_pct")))
        t["caf2"] = _mass(kg, d.get("CaF2_pct"))
        kpc = d.get("kg_per_charge")
        out[str(d.get("id", d.get("name", len(out))))] = dict(
            name=str(d.get("name", d.get("id", ""))), kg=_f(d.get("kg")), dry_kg=kg, comp=t, fe_kg=t["fe"],
            slag_kg=slag_forming_kg(t), kg_per_charge=(None if kpc is None else _f(kpc)), moist_pct=_pct(d.get("moist_pct")),
            pct={k: _pct(d.get(f"{k}_pct")) for k in ("SiO2", "Al2O3", "CaO", "MgO", "Fe", "Mn", "P", "S", "Ti", "Zn",
                                                     "Na2O", "K2O", "CaF2")},
            source=str(d.get("source", "") or ""))
    return out


def dust_components(du: Mapping[str, Any]) -> dict[str, float]:
    """The dust streams' components summed (:func:`dust_by_stream`)."""
    t = zero()
    for d in dust_by_stream(du).values():
        t = add(t, d["comp"])
    return t


def dust_kg(du: Mapping[str, Any]) -> float:
    """Total dust, kg/tHM (wet): the streams when the case carries them, else the pooled dc + gcp figures."""
    if du.get("streams") is not None:
        return float(sum(_f(d.get("kg")) for d in du["streams"] if d.get("enabled", True)))
    return _f(du.get("dc_kg")) + _f(du.get("gcp_kg"))


def dust_fe(du: Mapping[str, Any]) -> float:
    """Fe leaving with the dust, kg/tHM: the streams' analyses when the case carries them, else dc / gcp Fe %."""
    if du.get("streams") is not None:
        return float(dust_components(du)["fe"])
    return _f(du.get("dc_kg")) * _f(du.get("dc_Fe_pct")) / 100 + _f(du.get("gcp_kg")) * _f(du.get("gcp_Fe_pct")) / 100


# ---------------------------------------------------------------------------------------------- the balance
def balance(total: Mapping[str, float], dust: Mapping[str, float], r: Mapping[str, Any],
            hm: Mapping[str, float]) -> dict[str, Any]:
    """Page 9's sequence from the component totals (kg) to the slag (kg), verbatim.

    ``r``: the rules (``slag_balance`` of ``setting_bmo.yml``: Si_pct, C_pct, S_pct, other_pct held for the pig iron,
    pi_loss_pct, fe_to_pig_iron_fraction, mn_recovery_pct, sulphur_gas_loss_pct, alkali_to_slag_fraction,
    si_to_sio2_factor, fe_to_feo_factor, mn_to_mno_factor, slag_correction_factor, tio2_to_slag).
    ``hm``: the case's hot metal (Mn and Ti % reduced to the metal, page 9's ``mn_pct`` / ``ti_pct``)."""
    net = subtract_clamped(total, dust)
    pig_pct = max(1.0, 100.0 - _pct(r.get("C_pct")) - _pct(r.get("Si_pct")) - _pct(r.get("S_pct"))
                  - _pct(r.get("other_pct")))
    fe_frac = min(max(_f(r.get("fe_to_pig_iron_fraction")), 0.0), 1.0)
    fe_pi = net["fe"] * fe_frac
    fe_rem = max(0.0, net["fe"] - fe_pi)
    rec = _pct(r.get("mn_recovery_pct")) / 100.0
    mn_pi0, ti_pi0 = net["mn"] * rec, net["ti"] * rec
    th0 = (fe_pi + mn_pi0 + ti_pi0 + net["p"] + net["zn"]) * 100.0 / pig_pct
    loss = min(_pct(r.get("pi_loss_pct")), 99.0)
    act0 = th0 * (100.0 - loss) / 100.0
    hm_mn, hm_ti = _pct(hm.get("Mn")), _pct(hm.get("Ti"))
    mn_pi = min(net["mn"], act0 * hm_mn / 100.0) if hm_mn > 0 else mn_pi0
    ti_pi = min(net["ti"], act0 * hm_ti / 100.0) if hm_ti > 0 else ti_pi0
    mn_rem, ti_rem = max(0.0, net["mn"] - mn_pi), max(0.0, net["ti"] - ti_pi)
    metallic = fe_pi + mn_pi + ti_pi + net["p"] + net["zn"]
    theoretical = metallic * 100.0 / pig_pct
    actual = theoretical * (100.0 - loss) / 100.0
    sio2_used = theoretical * (_pct(r.get("Si_pct")) / 100.0) * max(0.0, _f(r.get("si_to_sio2_factor")))
    sio2 = max(0.0, net["sio2"] - sio2_used)
    feo = fe_rem * max(0.0, _f(r.get("fe_to_feo_factor")))
    mno = mn_rem * max(0.0, _f(r.get("mn_to_mno_factor")))
    s_pi = theoretical * (_pct(r.get("S_pct")) / 100.0)
    s_gas = net["s"] * (_pct(r.get("sulphur_gas_loss_pct")) / 100.0)
    s_slag = max(0.0, net["s"] - s_pi - s_gas)
    alk_frac = min(max(_f(r.get("alkali_to_slag_fraction")), 0.0), 1.0)
    alk = net["alkali"] * alk_frac
    tio2 = ti_rem / TI_IN_TIO2 if bool(r.get("tio2_to_slag", True)) else 0.0
    raw = dict(alk=alk, SiO2=sio2, Al2O3=net["al2o3"], CaO=net["cao"], MgO=net["mgo"], FeO=feo, MnO=mno, TiO2=tio2,
               S=s_slag, CaF2=net["caf2"])
    corr = max(0.0, _f(r.get("slag_correction_factor", 1.0)))
    comp = {k: v * corr for k, v in raw.items()}
    slag = float(sum(raw.values())) * corr
    return dict(slag_kg=slag, **comp, Fe_slag=fe_rem * corr, minor=comp["alk"] + comp["S"] + comp["CaF2"],
                B2=comp["CaO"] / comp["SiO2"] if comp["SiO2"] > 0 else float("inf"),
                Al2O3_pct=100 * comp["Al2O3"] / slag if slag > 0 else 0.0,
                MgO_pct=100 * comp["MgO"] / slag if slag > 0 else 0.0,
                net=net, pig_iron=dict(theoretical_kg=theoretical, actual_kg=actual, metal_pct=pig_pct,
                                       fe_kg=fe_pi, mn_kg=mn_pi, ti_kg=ti_pi, sio2_used_kg=sio2_used, s_kg=s_pi,
                                       s_gas_kg=s_gas, mn_rem_kg=mn_rem, ti_rem_kg=ti_rem,
                                       alk_gas_kg=net["alkali"] * (1.0 - alk_frac),
                                       tio2_left_out_kg=0.0 if bool(r.get("tio2_to_slag", True)) else ti_rem / TI_IN_TIO2))


#: The slag's components as ``balance`` names them, with the label page 9 prints.
SLAG_KEYS = (("CaO", "CaO"), ("SiO2", "SiO₂"), ("MgO", "MgO"), ("Al2O3", "Al₂O₃"), ("FeO", "FeO"), ("MnO", "MnO"),
             ("TiO2", "TiO₂"), ("S", "S"), ("alk", "Na₂O + K₂O"), ("CaF2", "CaF₂"))


def slag_detail(K: float, case: Mapping[str, Any], p: Mapping[str, Any], out: Mapping[str, Any],
                extra_lime: float = 0.0, extra_qtz: float = 0.0) -> dict[str, Any]:
    """The breakup page 9's *Slag Balance Details* shows, on the engine's case (kg/tHM):

    * ``by_material``: each burden row's kind, dry kg, Fe and slag-forming oxides (page 9's Fe (MT) and Slag (MT)
      columns per tHM) and components; ``by_fuel`` the same for coke / nut coke / PCI ash; ``dust`` per stream;
    * ``sources``: the components charged by ore, flux, fuel ash, and carried away by the dust; ``net`` after the dust;
    * ``removals``: what leaves the slag side for the hot metal and the gas (SiO2 reduced to Si, Fe / Mn / Ti / P / Zn
      to the metal, sulphur to the metal and the gas, alkalis to the gas, TiO2 left out on the workbook basis);
    * ``slag``: the final components, ``rollup``: slag-forming oxides per source, less the dust and the removals, equal
      to the slag."""
    r = p["slag_basis"]
    mats = burden_by_material(case)
    fuels = fuel_by_fuel(K, case)
    dusts = dust_by_stream(case["dust"])
    ore, flux = zero(), correction_flux_components(p, extra_lime, extra_qtz)
    for m in mats.values():
        if m["kind"] == "flux":
            flux = add(flux, m["comp"])
        else:
            ore = add(ore, m["comp"])
    fuel = zero()
    for f in fuels.values():
        fuel = add(fuel, f["comp"])
    dust = zero()
    for d in dusts.values():
        dust = add(dust, d["comp"])
    total = add(add(ore, flux), fuel)
    net = out["net"]
    pi = out["pig_iron"]
    removals = dict(SiO2_to_Si=pi["sio2_used_kg"], Fe_to_hot_metal=pi["fe_kg"],
                    MnO_to_hot_metal=pi["mn_kg"] * max(0.0, _f(r.get("mn_to_mno_factor"))),
                    TiO2_to_hot_metal=pi["ti_kg"] / TI_IN_TIO2, S_to_hot_metal=pi["s_kg"], S_to_gas=pi["s_gas_kg"],
                    alkali_to_gas=pi["alk_gas_kg"], P_to_hot_metal=net["p"], Zn_to_hot_metal=net["zn"],
                    TiO2_left_out=pi["tio2_left_out_kg"])
    slag = {k: _f(out.get(k)) for k, _ in SLAG_KEYS}
    # slag-forming oxides charged per source, less what the dust carries away, less what the hot metal and the gas take
    # (SiO2 reduced to Si, MnO / TiO2 reduced, alkalis to the gas; the residual, so the rows add up to the slag), plus
    # the FeO, S and CaF2 of the slag, which page 9's "slag-forming oxides" do not count
    rollup = dict(ore=slag_forming_kg(ore), flux=slag_forming_kg(flux), fuel_ash=slag_forming_kg(fuel),
                  dust=-(slag_forming_kg(total) - slag_forming_kg(net)), other=slag["FeO"] + slag["S"] + slag["CaF2"],
                  slag=_f(out.get("slag_kg")))
    rollup["hot_metal_gas"] = rollup["slag"] - (rollup["ore"] + rollup["flux"] + rollup["fuel_ash"] + rollup["dust"]
                                                + rollup["other"])
    return dict(by_material=mats, by_fuel=fuels, dust=dusts, sources=dict(ore=ore, flux=flux, fuel_ash=fuel, dust=dust),
                total=total, net=dict(net), removals=removals, slag=slag, rollup=rollup,
                pig_iron=dict(pi), rules=dict(Si_pct=_f(r.get("Si_pct")), S_pct=_f(r.get("S_pct")), C_pct=_f(r.get("C_pct")),
                                              tio2_to_slag=bool(r.get("tio2_to_slag", True)),
                                              alkali_to_slag_fraction=_f(r.get("alkali_to_slag_fraction")),
                                              sulphur_gas_loss_pct=_f(r.get("sulphur_gas_loss_pct")),
                                              fe_to_pig_iron_fraction=_f(r.get("fe_to_pig_iron_fraction")),
                                              slag_correction_factor=_f(r.get("slag_correction_factor", 1.0))),
                dust_kg=sum(d["kg"] for d in dusts.values()))


def slag_page9(K: float, case: Mapping[str, Any], p: Mapping[str, Any], extra_lime: float = 0.0,
               extra_qtz: float = 0.0) -> dict[str, Any]:
    """``bf_balance.slag_and_flux`` on page 9's basis: the case's burden, fuels and dust streams through
    :func:`balance` with the rules in ``p['slag_basis']``. Returns the keys ``slag_and_flux`` returns plus FeO,
    Fe_slag, alk, S, CaF2 (kg/tHM), ``minor``, the pig-iron closure and ``detail`` (:func:`slag_detail`)."""
    r = p["slag_basis"]
    total = add(burden_components(case, p, extra_lime, extra_qtz), fuel_components(K, case))
    out = balance(total, dust_components(case["dust"]), r, case["hm"])
    out.update(lime_kg=extra_lime, qtz_kg=extra_qtz, CO2_lime=extra_lime * p["lime_LOI"],
               CaO_lime=extra_lime * p["lime_CaO"], total=total)
    out["detail"] = slag_detail(K, case, p, out, extra_lime, extra_qtz)
    return out
