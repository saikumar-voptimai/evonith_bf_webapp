"""Sample times of the offline sources, for the page's freshness notices (read-only).

One query returns every valid lab record's time - raw-material chemistry, size analysis, strength, ash, dust and
burden-distribution changes - and a second returns the tonnes charged per material code per IST hour (the
materials in use). A record is VALID when the fields the model needs are filled: an ore row without Fe(T) is not a
chemistry result. Daily records (ash and dust analyses) are stamped at the END of their IST date.

The SQL is fixed in code (no user input), so it reads the tables directly rather than through the Data Explorer's
table whitelist.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

PLANT_TZ = "Asia/Kolkata"

_FINES_COLUMNS = ["plus_100mm", "plus_90mm", "plus_80mm", "plus_70mm", "plus_60mm", "plus_50mm", "plus_40mm", "plus_30mm",
                  "plus_25mm", "plus_20mm", "plus_16mm", "plus_12mm", "plus_10mm", "plus_8mm", "plus_6mm", "plus_5mm",
                  "minus_10mm", "minus_6mm", "minus_5mm"]


def _day_end(column: str) -> str:
    # end of the IST date, as an absolute time
    return f"(({column} + 1)::timestamp at time zone '{PLANT_TZ}')"


SAMPLES_SQL = f"""
select 'rm_chemistry' as source, material_code as material, date_time as t
  from offline_feed.ore_chemistry where fe_t is not null and date_time >= :start
union all
select 'rm_chemistry', material_code, date_time
  from offline_feed.sinter_chemistry where fe_t is not null and date_time >= :start
union all
select 'rm_chemistry', material_code, date_time
  from offline_feed.fuel_chemistry where ash is not null and date_time >= :start
union all
select 'rm_chemistry', material_code, date_time
  from offline_feed.flux_chemistry where coalesce(cao, mgo, sio2) is not null and date_time >= :start
union all
select 'fines', material_code, date_time
  from offline_feed.material_fines_analysis where coalesce({", ".join(_FINES_COLUMNS)}) is not null and date_time >= :start
union all
select 'rm_strength', material_code, date_time
  from offline_feed.raw_material_strength_analysis where property_1 is not null and date_time >= :start
union all
select 'ash_chemistry', material_type, {_day_end("date")}
  from offline_feed.ash_chemical_analysis where ash is not null and date >= (:start)::date
union all
select 'dust_basic', material_code, {_day_end("date")}
  from offline_feed.dust_basic_analysis where coalesce(fe_total, fc) is not null and date >= (:start)::date
union all
select 'dust_chemistry', material_code, {_day_end("date")}
  from offline_feed.dust_chemical_analysis where fe2o3 is not null
union all
select 'burden_distribution', '', date_time
  from ops_config.burden_history
"""

_CHARGE_CODES = ([f"sinter_{i}" for i in range(1, 5)] + ["pellet_1", "pellet_2"] + [f"ore_{i}" for i in range(1, 13)]
                 + ["flux_1", "flux_2", "flux_3", "coke_1", "coke_2", "nut_coke_1", "nut_coke_2"])

CHARGED_SQL = f"""
select date_trunc('hour', date_time at time zone '{PLANT_TZ}') as hour, count(*) as charges,
       {", ".join(f"sum({c}_mt) as {c}" for c in _CHARGE_CODES)}
  from offline_feed.charge_data
 where date_time >= :start
 group by 1
 order by 1
"""


def _engine():
    from furnace_data.relational.engine import build_relational_engine

    return build_relational_engine()


def fetch_offline_samples(start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(samples, charged): valid lab-record times since ``start`` (IST) and tonnes charged per hour per material."""

    start_utc = pd.Timestamp(start).tz_localize(PLANT_TZ).tz_convert("UTC").to_pydatetime()
    engine = _engine()
    try:
        with engine.connect() as con:
            samples = pd.read_sql(text(SAMPLES_SQL), con, params={"start": start_utc})
            charged = pd.read_sql(text(CHARGED_SQL), con, params={"start": start_utc})
    finally:
        engine.dispose()
    samples["t"] = pd.to_datetime(samples["t"], utc=True).dt.tz_convert(PLANT_TZ).dt.tz_localize(None)
    samples["material"] = samples["material"].fillna("").astype(str)
    charged["hour"] = pd.to_datetime(charged["hour"])
    charged = charged.set_index("hour").apply(pd.to_numeric, errors="coerce")
    return samples, charged


def material_names() -> dict[str, str]:
    """material_code -> readable name from plant_master.materials (e.g. ore_2 -> 'Lloyds CLO')."""

    engine = _engine()
    try:
        with engine.connect() as con:
            df = pd.read_sql(text("select material_code, material_name, material_description from plant_master.materials"), con)
    finally:
        engine.dispose()
    out = {}
    acronyms = {"clo": "CLO", "nmdc": "NMDC", "rom": "ROM", "pci": "PCI", "gcp": "GCP", "jrs": "JRS", "sp01": "SP-01",
                "sp02": "SP-02", "mn": "Mn"}
    for row in df.itertuples():
        # the plant's own short description where there is one (ore_2 'LLOYDS CLO'), else the material name
        desc = str(row.material_description or "").strip()
        usable = desc and len(desc) <= 30 and not desc.lower().startswith(("material slot", "canonical"))
        words = (desc if usable else str(row.material_name or row.material_code)).replace("_", " ").split()
        out[row.material_code] = " ".join(acronyms.get(w.lower(), w.capitalize()) for w in words)
    out.update({"coke": "Coke", "nutcoke": "Nut coke", "pci": "PCI", "dust_1": "Dust catcher dust", "dust_2": "GCP dust"})
    return out
