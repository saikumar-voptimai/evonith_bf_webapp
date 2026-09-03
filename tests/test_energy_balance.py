"""Energy balance: the convention, the terms, and the two costly mistakes.

The balance credits every reductant at full oxidation potential and books what
leaves unburnt as an output, so closure targets 1.00. Two errors during
development are pinned here so they cannot come back:

  * omitting iron oxide reduction, the largest term, put closure at 0.47
  * crediting carbon dissolved in hot metal as burnt over-stated input by
    ~1,400 MJ/tHM

Reference day is 2025-12-31 from docs/energy_balance_calculation_procedure.md.
"""

from __future__ import annotations

import pytest

from utils.energy_balance import (
    EnergyBalanceInputs,
    run_energy_balance,
    top_gas_volume_nm3_per_thm,
)
from utils.energy_balance.constants import hydrogen_pct_for_fuel, load_config


def _day() -> EnergyBalanceInputs:
    """The worked day, 2025-12-31."""

    return EnergyBalanceInputs(
        hot_metal_mt=2275.34,
        slag_mt=756.29,
        coke_mt=767.22,
        nut_coke_mt=171.48,
        pci_mt=342.63,
        blast_volume_nm3_per_hr=111392.03,
        blast_temperature_c=1200.0,
        oxygen_enrichment_pct=3.81,
        top_gas_co_pct=24.95,
        top_gas_co2_pct=18.75,
        top_gas_h2_pct=2.87,
        top_gas_temperature_c=137.26,
        hm_carbon_pct=4.28,
        hm_iron_pct=94.96,
        hm_silicon_pct=0.41,
        hm_manganese_pct=0.12,
        slag_feo_pct=0.36,
        flux_mt=9.5,
        ore_mt=1400.0,
        sinter_mt=2100.0,
        pellet_mt=380.0,
        flux_loi_pct=40.0,
        fuel_vm_pct={"coke": 0.9, "nut_coke": 1.0, "pci": 19.9},
        moisture_pct={"ore": 4.0, "pellet": 2.0, "flux": 1.0, "coke": 0.4,
                      "nut_coke": 8.5, "sinter": 0.0},
        shell_loss_gj_per_hr=28.29,  # stave rows 6-10 only - see the closure test
    )


# --- the convention -----------------------------------------------------------


def test_top_gas_volume_is_a_nitrogen_balance_with_no_fitted_constant():
    """N2 is inert, so V_top follows from measured quantities alone."""

    v = top_gas_volume_nm3_per_thm(
        blast_nm3_per_thm=1174.9,
        oxygen_enrichment_pct=3.81,
        co_pct=24.95,
        co2_pct=18.75,
        h2_pct=2.87,
    )
    # N2 blast 75.39%, N2 top 53.43% -> 1174.9 * 75.39 / 53.43
    assert v == pytest.approx(1174.9 * 75.39 / 53.43, rel=1e-3)
    # And it lands in the textbook 1,500-1,700 Nm3/tHM band.
    assert 1500.0 < v < 1750.0


def test_degraded_gas_analysis_returns_zero_rather_than_exploding():
    """A nonsense analysis would drive N2_top toward zero and V_top to infinity."""

    assert top_gas_volume_nm3_per_thm(
        blast_nm3_per_thm=1200.0, oxygen_enrichment_pct=4.0,
        co_pct=45.0, co2_pct=40.0, h2_pct=10.0,
    ) == 0.0


def test_balance_closes_on_the_reference_day():
    """UNRESOLVED: closure and the coke-rate solve want different shell losses.

    This fixture passes the STAVE ROWS 6-10 heat load, 28.29 GJ/hr -> 298
    MJ/tHM. The documented worked example instead uses the flow-scaled
    all-circuits figure, 886 MJ/tHM for this day, and that is where its headline
    closure of 1.002 comes from. The two have never agreed, and a +/-0.05
    tolerance here was wide enough that nobody noticed.

    Swapping this fixture to the flow-scaled value does fix closure - 0.967 ->
    1.004 - but then test_solver_reproduces_the_measured_coke_rate breaks,
    overshooting 337 kg/tHM by 11%. The two strongest validations of the balance
    disagree about the same number:

        shell loss    closure    solved coke rate
        298 MJ/tHM      0.967    339.7  (measured 337.2, +0.7%)
        886 MJ/tHM      1.004    375.2  (+11.3%)

    Neither is obviously right. A BF should lose 200-400 MJ/tHM through the
    shell, which favours 298; but 298 leaves closure 3% short. The flow-scaled
    figure assumes every cooling circuit runs the same temperature rise, which
    is the weakest link and worth measuring per circuit.

    Pinned at the stave value so the solver validation stands, and asserted
    loosely on purpose - tightening it would force a choice the data does not
    yet support.
    """

    result = run_energy_balance(_day())

    assert result.closure == pytest.approx(0.967, abs=0.01)
    assert result.closure_band in {"green", "amber"}


# --- the two mistakes ---------------------------------------------------------


def test_iron_reduction_is_the_largest_single_term():
    """Omitting it once put closure at 0.47. It is ~45% of the output side."""

    result = run_energy_balance(_day())
    largest = max(result.demand, key=lambda k: result.demand[k])

    assert largest == "iron_reduction"
    assert result.demand["iron_reduction"] == pytest.approx(7000.0, rel=0.05)
    assert result.demand["iron_reduction"] > 0.4 * result.total_output_mj_per_thm


def test_carbon_dissolved_in_hot_metal_is_not_credited_as_fuel():
    """It never burns. Crediting it over-states input by ~1,400 MJ/tHM."""

    day = _day()
    result = run_energy_balance(day)
    d = result.diagnostics

    assert d["carbon_to_hot_metal_kg_per_thm"] == pytest.approx(42.8, rel=1e-2)
    assert d["carbon_burnt_kg_per_thm"] == pytest.approx(
        d["carbon_charged_kg_per_thm"] - d["carbon_to_hot_metal_kg_per_thm"]
    )

    # Pretend it does burn, and watch the input inflate by the known amount.
    inflated = run_energy_balance(
        EnergyBalanceInputs(**{**day.__dict__, "hm_carbon_pct": 0.0})
    )
    delta = inflated.total_input_mj_per_thm - result.total_input_mj_per_thm
    assert delta == pytest.approx(42.8 * 32.8, rel=0.02)
    assert delta > 1300.0


# --- hydrogen -----------------------------------------------------------------


def test_shipped_hydrogen_values_match_the_coals_rank():
    """No ultimate analysis exists or is coming, so these come from the rank.

    The plant's PCI is medium-volatile bituminous (22.4% VM daf), which runs
    4.5-4.9% H daf, i.e. ~4.2% as charged. Coke is 0.3-0.5%. A value outside
    those bands means someone has substituted a guess for the rank data, so the
    bounds are asserted rather than the exact figures.
    """

    h_pci, source_pci = hydrogen_pct_for_fuel("pci", 20.03)
    h_coke, source_coke = hydrogen_pct_for_fuel("coke", 0.94)

    assert 3.5 <= h_pci <= 5.0
    assert 0.25 <= h_coke <= 0.55
    # Provenance must never claim more authority than a literature value has.
    assert "configured" in source_pci and "configured" in source_coke


def test_the_vm_backstop_reproduces_the_shipped_values():
    """It only fires when a hydrogen_pct is null, and must not contradict it.

    The previous flat factor of 0.25 returned 5.0% H for PCI at VM 20, above any
    real medium-volatile bituminous coal. That overstatement was for years read
    as evidence the hydrogen term could not be trusted.
    """

    cfg = load_config()
    nulled = {**cfg, "fuels": {**cfg["fuels"], "hydrogen_pct": {"pci": None}}}

    h_pct, source = hydrogen_pct_for_fuel("pci", 20.03, nulled)

    assert "estimated" in source
    assert h_pct == pytest.approx(hydrogen_pct_for_fuel("pci", 20.03)[0], abs=0.25)


def test_configured_hydrogen_overrides_the_correlation():
    cfg = load_config()
    cfg = {**cfg, "fuels": {**cfg["fuels"], "hydrogen_pct": {"pci": 4.5}}}

    h_pct, source = hydrogen_pct_for_fuel("pci", 19.9, cfg)

    assert h_pct == pytest.approx(4.5)
    assert "configured" in source


def test_fuel_hydrogen_is_computed_but_off_by_default():
    """Physically right, deliberately not credited - and this is now MEASURED.

    The old reason was an unknown H%. That is settled: H% is fixed from the
    coal's rank. The term stays off because the residual it would fill scales
    with total fuel, not with hydrogen - over a PCI range of 152-205 kg/tHM the
    correlation with the back-calculated residual is -0.05. Enabling it costs
    ~6 points of closure and buys nothing.

    See scripts/pci_hydrogen_from_closure.py. Do not flip this default without
    re-running it.
    """

    result = run_energy_balance(_day())
    d = result.diagnostics

    assert d["fuel_hydrogen_included"] is False
    assert result.supply["hydrogen"] == 0.0
    # Still reported, so the size of what is being withheld stays visible.
    assert d["fuel_hydrogen_mj_per_thm_if_included"] > 500.0


def test_enabling_hydrogen_moves_closure_by_the_reported_amount():
    """No hidden coupling: turning it on shifts input by exactly the figure
    the diagnostics advertise."""

    day = _day()
    cfg = load_config()
    on = {**cfg, "supply": {**cfg["supply"], "include_fuel_hydrogen": True}}

    off_result = run_energy_balance(day)
    on_result = run_energy_balance(day, on)

    delta = on_result.total_input_mj_per_thm - off_result.total_input_mj_per_thm
    assert delta == pytest.approx(
        off_result.diagnostics["fuel_hydrogen_mj_per_thm_if_included"]
    )
    assert on_result.closure < off_result.closure


def test_blast_moisture_is_never_credited_as_fuel():
    """It arrives already oxidised. Crediting it would invent energy."""

    prov = run_energy_balance(_day()).diagnostics["hydrogen_provenance"]

    assert prov["blast_moisture"]["credited_as_fuel"] is False
    assert prov["blast_moisture"]["kg_h2o_per_thm"] > 0.0
    # PCI dominates fuel hydrogen: ~6.3 kg H/tHM against ~2 from blast moisture.
    assert prov["pci"]["kg_h_per_thm"] > 3.0 * prov["blast_moisture"]["kg_h_per_thm"]


# --- structure ----------------------------------------------------------------


def test_top_gas_volume_is_cross_checked_against_a_carbon_balance():
    """Two independent volumes that must agree - and on this plant, do not.

    Carbon gives ~14% more top gas than nitrogen. The disagreement tracks the
    balance's residual (r = +0.77) and drives most of its across-quarter drift.
    Root cause is the gas analysis both are computed from: an under-read of
    CO+CO2 pushes the two estimates in opposite directions.

    Reported rather than corrected, so the flag must actually fire on a typical
    day. If this test starts failing because the ratio came back to 1.0, the
    analyser has been fixed - which is the outcome we want, not a broken test.
    """

    d = run_energy_balance(_day()).diagnostics

    assert d["top_gas_nm3_per_thm_carbon_basis"] > d["top_gas_nm3_per_thm"]
    assert d["top_gas_volume_ratio_carbon_to_nitrogen"] == pytest.approx(1.14, abs=0.06)
    assert d["gas_analysis_suspect"] is True


def test_carbon_basis_volume_agrees_when_the_analysis_is_consistent():
    """The check must not fire on good data, or it is just noise.

    Raising CO+CO2 by the ~3 points the reconciliation calls for should bring
    the two volumes together and clear the flag.
    """

    day = _day()
    fixed = EnergyBalanceInputs(
        **{**day.__dict__, "top_gas_co_pct": 26.7, "top_gas_co2_pct": 20.1}
    )

    d = run_energy_balance(fixed).diagnostics

    assert d["top_gas_volume_ratio_carbon_to_nitrogen"] == pytest.approx(1.0, abs=0.05)
    assert d["gas_analysis_suspect"] is False


def test_implied_shell_loss_is_reported_for_cross_checking():
    """If it disagrees with the measured value, a term is missing.

    Calling the residual 'shell loss' regardless would hide the error, so the
    two numbers are always reported side by side.
    """

    day = _day()
    result = run_energy_balance(day)
    measured = result.demand["shell_loss"]

    assert measured > 0.0
    assert result.implied_shell_loss_mj_per_thm != pytest.approx(measured, rel=1e-6)
    # Both should at least be the same order of magnitude on a sane day.
    assert 0.1 < result.implied_shell_loss_mj_per_thm / measured < 10.0


def test_zero_hot_metal_is_rejected_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="hot_metal_mt"):
        run_energy_balance(EnergyBalanceInputs(**{**_day().__dict__, "hot_metal_mt": 0.0}))


def test_totals_are_the_sum_of_their_parts():
    result = run_energy_balance(_day())

    assert result.total_demand_mj_per_thm == pytest.approx(sum(result.demand.values()))
    assert result.total_top_gas_mj_per_thm == pytest.approx(sum(result.top_gas.values()))
    assert result.total_input_mj_per_thm == pytest.approx(sum(result.supply.values()))
    assert result.total_output_mj_per_thm == pytest.approx(
        result.total_demand_mj_per_thm + result.total_top_gas_mj_per_thm
    )


# --- deriving control coefficients from the closed balance --------------------


def test_solver_reproduces_the_measured_coke_rate():
    """The strongest single validation of the whole balance.

    Nothing in the solve is told what the coke rate was. It falls out of
    requiring the balance to close, so agreeing with the charge reports means
    the physics and the constants are both right.
    """

    from utils.energy_balance.solve import solve_coke_rate_kg_per_thm

    day = _day()
    measured = day.coke_mt / day.hot_metal_mt * 1000.0
    solved = solve_coke_rate_kg_per_thm(day)

    assert solved == pytest.approx(measured, rel=0.05)


def test_blast_temperature_coefficient_matches_the_plant_figure():
    """The Phase 2 gate.

    A naive one-line formula gives -5.8 kg per 100 C because it holds the top
    gas fixed. Carrying the carbon balance so top gas shrinks with coke gives
    about -10, which is what the plant's own config says. Independent agreement
    between derived physics and plant practice.
    """

    from utils.energy_balance.solve import derive_control_coefficients

    coeff = derive_control_coefficients(_day())["blast_temperature_c"]

    assert -12.0 <= coeff["reported_value"] <= -8.0
    # And decisively away from the naive figure that ignores the feedback.
    assert coeff["reported_value"] < -7.0


def test_top_pressure_is_reported_as_not_derivable():
    """It acts through eta_CO, which the solve holds fixed.

    Returning a number here would be fabricating one.
    """

    from utils.energy_balance.solve import derive_control_coefficients

    entry = derive_control_coefficients(_day())["top_pressure_bar"]

    assert entry["derivative_kg_coke_per_unit"] is None
    assert "eta_CO" in entry["note"]


def test_pci_replacement_from_the_balance_is_carbon_equivalence_not_plant_practice():
    """A trap worth pinning.

    The balance returns 0.75/0.87 = 0.86 kg coke per kg PCI, because on pure
    energy terms that is the carbon ratio. The plant uses 0.53, which is lower
    because coke also holds the burden column open and PCI cannot. The balance
    cannot see that mechanical role, so its number must NOT replace 0.53.
    """

    from utils.energy_balance.solve import derive_control_coefficients

    entry = derive_control_coefficients(_day())["pci_mt"]

    assert entry["reported_value"] == pytest.approx(0.75 / 0.87, rel=0.05)
    assert entry["reported_value"] > 0.53 * 1.4
    assert "0.53" in entry["benchmark"]


def test_inconsistent_inputs_raise_rather_than_returning_nonsense():
    from utils.energy_balance.solve import solve_coke_rate_kg_per_thm

    absurd = EnergyBalanceInputs(**{**_day().__dict__, "hm_iron_pct": 5000.0})
    with pytest.raises(ValueError, match="closes the balance"):
        solve_coke_rate_kg_per_thm(absurd)
