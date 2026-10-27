"""Imported by the forkserver of the STRATA-Flow solver processes (SV-1.8, ``strata_network.start_solvers``).

It builds the STRATA-Flow context (burden builder, calibrated cohesive zone; about 2 s) once in the forkserver, so
that every solver process forked from it starts with the context built: the first pool of a process pays the build,
later pools of the same process start in a fraction of a second. It never raises; on a failure each solver process
builds its own context as before.
"""

from __future__ import annotations

try:
    from utils.strata_flow.omega_eval import context as _context

    _context()
except Exception:  # noqa: BLE001 - the forkserver must start in any case
    pass
