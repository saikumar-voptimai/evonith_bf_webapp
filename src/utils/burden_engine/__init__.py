"""BF-02 burden engine: blend + flux + production optimisation on the two-zone (Rist) energy model.

Pure numerical code (no Streamlit, no I/O beyond reading JSON inputs). Used by the Burden & Fuel Optimiser page.

    from utils.burden_engine import Engine, Settings, Limits
    eng = Engine(inputs, cal)                       # inputs: yard / prices / limits; cal: two-zone calibration
    r = eng.min_cost(prod_t_day=2270, fuel=True)    # Step 2: fixed production, ore + flux + fuel
    r = eng.max_production(Limits(process=True))    # highest output within the day's limits
    r = eng.min_total_cost(Limits(process=True))    # Step 1: cheapest production level, with the frontier

Coefficients outside the balance come from the fact register (``src/config/facts_register.json``).
"""

from utils.burden_engine.engine import (BandError, Engine, Infeasible, Limits, Settings, summary,
                                        value_in_use)
from utils.burden_engine.facts import FactRegister, RetiredFactError, load_register

__all__ = ["BandError", "Engine", "FactRegister", "Infeasible", "Limits", "RetiredFactError", "Settings",
           "load_register", "summary", "value_in_use"]
