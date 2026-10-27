"""Gate of the HMT forecast (SV-1.4, #284): model against persistence one descent time ahead.

    python scripts/hmt_gate.py --from 2026-01-01 --to 2026-10-08 [--test "2026 expanding"] [--fact COKE-18]
                               [--out forecasts.csv] [--write]

Loads the Coke Energy sources as page 10 does (hourly dataset, 10-min InfluxDB data, Neon taps and DPR), runs the
pipeline once and evaluates every hourly window of the period (``utils.coke_energy.twin.evaluate_window`` without
STRATA): HMT(t), K_arr(t), K_FORGE at HMT(t), its HMT slope and HMT_f(t + tau_RW). The target is the tap temperature
at t + tau_RW. Prints the gate (RMSE at least 10 % below persistence, |bias| <= 5 degC; ``twin.hmt_forecast.gate`` in
``config/coke_energy.yml``) and with ``--write`` records it as a fact of the register (COKE-18: 2026 expanding test,
COKE-19: 2023-24 test with frozen parameters; Validated on a pass, Disputed on a fail) with a revision-log entry.
The forecast has no fitted parameter: the 14-day calibration of FORGE is out of sample by construction, so the
2026 run is the expanding test, and a run on the 2023-24 sources is the frozen-parameter test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "furnace_data")]
REGISTER = ROOT / "src" / "config" / "facts_register.json"


def forecasts(frames, pool, settings, start, end, transit_ref) -> list[dict]:
    import pandas as pd

    from utils.coke_energy import hmt_forecast
    from utils.coke_energy.twin import Window, evaluate_window

    temps = hmt_forecast.taps(frames.hm)
    rows = []
    for t in pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="h", inclusive="left"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = evaluate_window(
                frames.fd,
                frames.m10,
                frames.hm,
                frames.dpr,
                Window.ending(t),
                pool,
                settings,
                frames.provisional,
                strata=False,
                transit=transit_ref,
            )
        q = ev.quantities
        f, h = q.get("hmt_forecast_c"), q.get("hmt_forecast_horizon_h")
        rows.append(
            dict(
                t=t,
                status=ev.status,
                hmt_now=q.get("hmt_now_c"),
                hmt_forecast=f,
                horizon_h=h,
                k_arr=q.get("k_arr_kg_thm"),
                k_forge_hmt_now=q.get("k_forge_hmt_now_kg_thm"),
                dk_dhmt=q.get("dk_dhmt_kg_thm_per_c"),
                hmt_target=hmt_forecast.target(temps, t + pd.Timedelta(hours=h)) if h is not None else None,
            )
        )
    return rows


def write_fact(fact: dict, change: str, path: Path = REGISTER) -> str:
    """Insert or replace ``fact`` in the register, bump the minor version and log the change."""
    reg = json.loads(path.read_text(encoding="utf-8"))
    major, minor = (int(x) for x in reg["version"].split("."))
    version = f"{major}.{minor + 1}"
    fact = dict(fact, version=version)
    reg["facts"] = [f for f in reg["facts"] if f["id"] != fact["id"]] + [fact]
    reg["revision_log"].insert(0, dict(version=version, date=fact["date"], change=change))
    reg["version"], reg["date"] = version, fact["date"]
    path.write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return version


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from", dest="start", required=True, help="first window end (IST)")
    ap.add_argument("--to", dest="end", required=True, help="last window end (exclusive, IST)")
    ap.add_argument("--test", default="2026 expanding test", help="name of the test in the register")
    ap.add_argument("--fact", default="COKE-18", help="register id to record (COKE-18 2026, COKE-19 2023-24)")
    ap.add_argument("--out", type=Path, help="write the forecasts as CSV")
    ap.add_argument("--write", action="store_true", help="record the gate result in the fact register")
    ap.add_argument("--register", type=Path, default=REGISTER, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    import pandas as pd

    from services import coke_energy as ce
    from services import transit
    from services.twin import load_frames
    from utils.coke_energy import hmt_forecast
    from utils.coke_energy.pipeline import PipelineSettings

    cfg = ce.config()
    settings = PipelineSettings.from_config(cfg)
    keys = ce.data_keys(cfg, 0)
    t0 = time.perf_counter()
    frames = load_frames(keys)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pool = ce.page_data(keys, settings).result
    ref, _ = transit.settings(cfg)
    rows = forecasts(frames, pool, settings, args.start, args.end, ref)
    df = pd.DataFrame(rows)
    criteria = hmt_forecast.GateCriteria.from_config(((cfg.get("twin") or {}).get("hmt_forecast") or {}).get("gate"))
    result = hmt_forecast.gate(df, criteria)
    with_target = df.dropna(subset=["hmt_forecast", "hmt_target"])
    print(
        f"{len(df)} hourly windows {args.start} to {args.end} ({time.perf_counter() - t0:.0f} s); "
        f"{len(with_target)} forecasts with a target"
    )
    if len(with_target):
        print(
            f"  horizon median {with_target.horizon_h.median():.2f} h; slope dK/dHMT median "
            f"{with_target.dk_dhmt.median():.3f} kg/tHM per degC; K_arr - K_FORGE(HMT) SD "
            f"{(with_target.k_arr - with_target.k_forge_hmt_now).std():.1f} kg/tHM"
        )
    print(f"gate ({args.test}): {'PASS' if result.passed else 'FAIL'}: {result.reason}")
    if args.out:
        df.to_csv(args.out, index=False)
    if args.write:
        today = date.today().isoformat()
        fact = hmt_forecast.register_fact(
            result, args.fact, args.test, f"{args.start} to {args.end}", criteria, today, ""
        )
        version = write_fact(
            fact,
            f"SV-1.4: HMT forecast gate, {args.test} ({args.fact} {'Validated' if result.passed else 'Disputed'}).",
            args.register,
        )
        print(f"recorded {args.fact} in {args.register.name} v{version}")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
