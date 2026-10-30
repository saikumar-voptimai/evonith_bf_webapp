"""Live inputs of the optimisers (SV-7.3, #326): stock, raw-material composition, process conditions and the operator
inputs in force, read once per TTL and served with the time of their data.

``GET /api/inputs/live`` is ``services.live_inputs.snapshot``: the four groups in one response with a ``snapshot_id``
(hash of the four as-of times), ``read_at``, and per group ``as_of``, ``stale``, ``source``, the values with their
units and the group's TTL and stale limit from ``config/live_inputs.yml`` (the SPA holds no copy). A run of the Blend
Mix Optimiser or the Burden & Fuel Optimiser stores the id of the snapshot it was built from, so its inputs can be
matched with the panel's values. Any number of readers within a group's TTL cause one upstream read.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from api.auth import require_role
from services import live_inputs
from services.live_inputs import LiveInputsView

router = APIRouter(prefix="/api/inputs", tags=["inputs"], dependencies=[Depends(require_role("user"))])


@router.get("/live", response_model=LiveInputsView, summary="Live inputs of the optimisers")
def live(
    chemistry_mode: Annotated[
        Literal["latest", "avg"] | None,
        Query(
            description=(
                "Chemistry snapshot of the composition group: latest (the sample used before the latest charge) or "
                "avg (the window's average); omitted: page 9's setting (setting_bmo.yml)"
            )
        ),
    ] = None,
    chemistry_window_days: Annotated[
        int | None, Query(ge=1, le=365, description="Chemistry window, d; omitted: page 9's setting")
    ] = None,
) -> LiveInputsView:
    """Stock (latest entry per material), raw-material composition (page 9's chemistry snapshot), process conditions
    (the furnace-status snapshot's blast, oxygen, steam, PCI, fuel-rate, top-gas, RAFT and production readings with
    the plan day's blast moisture) and the operator inputs in force, each with its as-of time, stale flag, source and
    limits; ``snapshot_id`` hashes the four as-of times."""
    return live_inputs.snapshot(chemistry_mode=chemistry_mode, chemistry_window_days=chemistry_window_days)
