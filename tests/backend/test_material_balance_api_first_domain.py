from __future__ import annotations

from datetime import date

import pandas as pd

from furnace_data.material_balance.data_sources import resolve_material_balance_windows
from furnace_data.material_balance.dpr_mapping import apply_dpr_mapping


def test_hour_lag_windows_shift_start_and_end_in_ist_and_utc():
    output, raw_material, blast = resolve_material_balance_windows(
        date(2026, 7, 22),
        rm_lag_hours=96,
        blast_lag_hours=4,
    )

    assert output.local_start.isoformat() == "2026-07-22T00:00:00+05:30"
    assert output.utc_start.isoformat() == "2026-07-21T18:30:00+00:00"
    assert raw_material.local_start.isoformat() == "2026-07-18T00:00:00+05:30"
    assert raw_material.local_end.isoformat() == "2026-07-19T00:00:00+05:30"
    assert blast.local_start.isoformat() == "2026-07-21T20:00:00+05:30"
    assert blast.local_end.isoformat() == "2026-07-22T20:00:00+05:30"


def test_dpr_mapping_uses_latest_non_null_snapshot_not_sum():
    dpr = pd.DataFrame({"coke_total_mt": [100.0, None, 125.0]})

    result = apply_dpr_mapping(dpr, {"coke_mass_t": "coke_total_mt"})

    assert result["coke_mass_t"] == 125.0