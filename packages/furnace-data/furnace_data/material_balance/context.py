"""Material Balance context construction and runtime config checks."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any

from furnace_data.material_balance.data_sources import (
    WINDOW_POLICY_VERSION,
    aggregate_hm_slag_from_snapshot,
    aggregate_online_from_snapshot,
    aggregate_rm_from_snapshot,
    fetch_dpr_for_window,
    load_static_dataset_snapshot,
    resolve_material_balance_windows,
)
from furnace_data.material_balance.dpr_mapping import load_full_config
from furnace_data.material_balance.types import MaterialBalanceContext

ALGORITHM_VERSION = "legacy_v1"
CATALOG_VERSION = "material-balance-catalog-v1"


def config_checksum(config: dict[str, Any]) -> str:
    """Return a stable config checksum used as a fallback version."""

    payload = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def effective_config_version(config: dict[str, Any]) -> str:
    raw = str(config.get("version") or config.get("config_version") or "").strip()
    return raw or f"mbcfg-{config_checksum(config)}"


class MaterialBalanceContextBuilder:
    """Acquire all external Material Balance inputs exactly once per run."""

    def __init__(self, *, config: dict[str, Any] | None = None) -> None:
        self.config = config

    def build(
        self,
        *,
        day: date,
        rm_lag_hours: int = 0,
        blast_lag_hours: int = 0,
        dust_catcher_t: float = 0.0,
        algorithm_version: str = ALGORITHM_VERSION,
    ) -> MaterialBalanceContext:
        cfg = dict(self.config if self.config is not None else load_full_config())
        output_window, rm_window, blast_window = resolve_material_balance_windows(
            day,
            rm_lag_hours=int(rm_lag_hours),
            blast_lag_hours=int(blast_lag_hours),
        )
        snapshot = load_static_dataset_snapshot()
        rm_df = aggregate_rm_from_snapshot(snapshot, rm_window)
        hm_slag_df = aggregate_hm_slag_from_snapshot(snapshot, output_window)
        online = aggregate_online_from_snapshot(snapshot, blast_window)
        dpr_df = fetch_dpr_for_window(output_window)
        dpr_mapping = {
            key: value
            for key, value in (cfg.get("dpr_field_mapping") or {}).items()
        }
        data_quality = {
            "raw_material_rows": int(rm_df.attrs.get("n_rows", 0)) if rm_df is not None else 0,
            "raw_material_expected_rows": 24,
            "raw_material_coverage_pct": _coverage(rm_df.attrs.get("n_rows", 0) if rm_df is not None else 0),
            "hot_metal_slag_rows": int(hm_slag_df.attrs.get("n_rows", 0)) if hm_slag_df is not None else 0,
            "hot_metal_slag_expected_rows": 24,
            "hot_metal_slag_coverage_pct": _coverage(hm_slag_df.attrs.get("n_rows", 0) if hm_slag_df is not None else 0),
            "process_rows": _estimate_process_rows(online),
            "process_expected_rows": 24,
            "process_coverage_pct": 100.0 if online else 0.0,
            "dpr_rows": int(len(dpr_df)) if dpr_df is not None else 0,
            "dpr_expected_rows": 1,
            "dpr_coverage_pct": 100.0 if dpr_df is not None and not dpr_df.empty else 0.0,
        }
        return MaterialBalanceContext(
            day=day,
            config=cfg,
            config_version=effective_config_version(cfg),
            dataset_snapshot=snapshot,
            output_window=output_window,
            raw_material_window=rm_window,
            blast_window=blast_window,
            rm_df=rm_df,
            hm_slag_df=hm_slag_df,
            online=online,
            dpr_df=dpr_df,
            dpr_mapping=dpr_mapping,
            rm_lag_hours=int(rm_lag_hours),
            blast_lag_hours=int(blast_lag_hours),
            dust_catcher_t=float(dust_catcher_t),
            algorithm_version=algorithm_version,
            window_policy_version=WINDOW_POLICY_VERSION,
            data_quality=data_quality,
        )


def _coverage(rows: Any) -> float:
    try:
        return round(min(max(float(rows), 0.0), 24.0) / 24.0 * 100.0, 1)
    except (TypeError, ValueError):
        return 0.0


def _estimate_process_rows(online: dict[str, float]) -> int:
    return 24 if online else 0