"""STRATA-Flow (vendored bf2_momentum v0.4.1): layered, radially resolved pressure-loss (momentum) model of BF-02.

Copied into the webapp for STRATA 1.3 so that the burden engine solves the network itself (no surrogate); see
``utils/burden_engine/strata_network.py``. Changes against bf2_momentum v0.4: plant RI / RDI catalogue (23 Oct 2026).
Not carried: demo.py, figures.py, run_sector_scenarios.py, sectors.py, moisture.py and the leak modules (other models).

What it is
----------
A resistance-network model of the gas flow through the burden column: the furnace is divided into
rings (radial) and levels (axial); every cell carries the ore and coke layers that the charging
programme put there, with porosity derived from the measured size distributions through a packing
model; the cohesive zone is a separate resistance (coke slits between fused ore layers); the
dripping zone and the deadman carry liquid hold-up and fines. The network is solved for the
pressure field and the gas flow through every cell, which gives

* the blast pressure needed for a blast volume (or the volume a pressure limit allows), i.e. the
  bed resistance K of the register's packed-bed law, now derived rather than fitted;
* the radial split of the gas between the centre and the wall at every level, and the shape of
  the cohesive zone that goes with it;
* a contact-based estimate of the shaft efficiency ω (hypothesis, to be tested against the
  14-day calibration record).

It is NOT a discrete-element model: it does not track particles. It is the continuum model that a
DEM would be used to calibrate (layer porosity, mixed-layer thickness, fines percolation).

Modules: geometry (drawn BF-02 profile), plant (stave rows, cooling circuits, tuyeres, 110 thermocouples, stave
log), materials, packing, gas, ergun, charging, cohesive, network, kequiv, omega, heatload (heat-load tags and
stave TCs → model heights), transit (descent time, coke inventory), sectors (circumferential sectors in parallel),
plant_effects (leak water and cooling water for FORGE/TRACE), reference, run_case, demo, figures.
"""

from .reference import reference_case, run_reference  # noqa: F401
from .network import solve_network, NetworkResult  # noqa: F401
from .kequiv import k_equivalent, register_sensitivities  # noqa: F401

__version__ = "0.2.0"   # 21 Oct 2026: drawn geometry, plant layout, sectors
