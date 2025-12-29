# pages/ai_copilot.py
import os
import io
import json
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from utils.helper_functions_explorer import data_retrieval as dr
from utils.anomaly_propensity import compute_propensity_suite
from config.config_loader import load_config
from dotenv import load_dotenv
from openai import OpenAI
client = OpenAI()

load_dotenv()

config = load_config("setting_ds_dv.yml")  # Load the configuration file
config_vsense = load_config('setting_vsense.yml')


# ────────────────────────────────────────────────────────────────────────────────
# 0) CONFIG
# ────────────────────────────────────────────────────────────────────────────────

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Prefer an env-provided name; fall back to a safe small model
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-5-mini")
USE_CODE_INTERPRETER = True 

# Local CSV for static fuel/ETA analyses
STATIC_CSV_PATH = config['DATA']

model_keys = list(config_vsense['Optimisation'].keys())
CONTROL_COLUMNS = config_vsense['Optimisation'][model_keys[1]]['control_params']
INPUT_COLUMNS = [item for sublist in config_vsense['Optimisation'][model_keys[1]]['input_params'].values() for item in sublist]
OUTPUT_COLUMNS = [config_vsense['Optimisation'][key]['output_param'] for key in config_vsense['Optimisation']]

METRIC_MAP = {
    "ETA CO": "FurnaceTopGasAnalysisCO2ETACO",
    "Total Fuel": "Act. Fuel RateKg/Thm.",
}

ALL_COLUMNS = CONTROL_COLUMNS + INPUT_COLUMNS + OUTPUT_COLUMNS
BEST_START = datetime(2024, 4, 1, tzinfo=timezone.utc)
BEST_END   = datetime(2024, 6, 30, 23, 59, 59, tzinfo=timezone.utc)


# ────────────────────────────────────────────────────────────────────────────────
# 1) LLM WRAPPER (OpenAI Responses API with optional code_interpreter)
# ────────────────────────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str, files: list[tuple[str, bytes]] | None = None) -> str:
    """
    Sends prompts to OpenAI Responses API.
    - Optionally enables the code_interpreter tool.
    - If files are provided, uploads and attaches them for the tool to access.
    - Falls back to a plain Chat Completions request if Responses API call fails.
    """
    if not OPENAI_API_KEY:
        return "⚠️ OPENAI_API_KEY not set."

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Upload files and collect their IDs
    file_ids: list[str] = []
    if files:
        for fname, fbytes in files:
            try:
                up = client.files.create(file=(fname, io.BytesIO(fbytes)), purpose="assistants")
                file_ids.append(up.id)
            except Exception:
                # ignore upload errors, proceed without that file
                pass

    # Build request
    req = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    tools = []
    if USE_CODE_INTERPRETER:
        tools.append({"type": "code_interpreter", "container": {"type": "auto"}})
    if tools:
        req["tools"] = tools

    # Attach files primarily via top-level attachments, fallback via tool_resources
    if file_ids:
        req["attachments"] = [
            {"file_id": fid, "tools": [{"type": "code_interpreter"}]} for fid in file_ids
        ]
        req["tool_resources"] = {"code_interpreter": {"file_ids": file_ids}}

    # Try Responses API first
    last_err = None
    try:
        response = client.responses.create(**req)
        # New SDK provides output_text for convenience
        text = getattr(response, "output_text", None)
        if text:
            return text
        # Fallback to raw JSON dump if no convenience field is present
        return json.dumps(response.to_dict(), indent=2)
    except Exception as err1:
        last_err = err1

    return f"LLM call failed: {last_err}"
    
# ────────────────────────────────────────────────────────────────────────────────
# 2) CSV LOADER (STATIC ANALYSIS: Review / Drivers)
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_static_df(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        st.warning(f"CSV not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df['Unnamed: 0'], utc=True, errors="coerce")
    df.drop(columns=['Unnamed: 0'], inplace=True, errors="ignore")
    # Ensure numeric
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def best_snapshot_from_static(df: pd.DataFrame, target_col: str):
    """
    Returns the best snapshot for a target column in the static CSV.
    The best snapshot is defined as the time when the target column is at its best value
    within the BEST_START and BEST_END range.
    Args:
        df (pd.DataFrame): DataFrame containing the static data.
        target_col (str): The target column to find the best snapshot for.
    Returns:
        dict: A dictionary with the best time and value for the target column.
    """
    if df.empty or target_col not in df.columns:
        return None
    period = df.loc[(df.index >= BEST_START) & (df.index <= BEST_END), [target_col]].dropna()
    if period.empty:
        return None
    # ETA CO → max; Total Fuel → min
    if target_col == "FurnaceTopGasAnalysisCO2ETACO":
        ts = period[target_col].idxmax()
    else:
        ts = period[target_col].idxmin()
    return {"when": ts, "value": float(period.loc[ts, target_col])}


# ────────────────────────────────────────────────────────────────────────────────
# 3) INFLUX HELPERS (ONLINE ANOMALIES)
# ────────────────────────────────────────────────────────────────────────────────
MEASUREMENT_LABELS = {
    "heatload_delta_t": "Heatload Delta T",
    "process_params": "Process Params",
    "temperature_profile": "Temperature Profile"
}

FREQUENCY_TO_TIMEDTA = {
    "None": None,
    "1 minute": "1min",
    "5 minutes": "5min",
    "10 minutes": "10min",
    "15 minutes": "15min",
    "30 minutes": "30min",
    "1 hour": "1h",
    "6 hours": "6h",
    "8 hours": "8h",
    "12 hours": "12h",
    "1 day": "1d",
}

FIELD_LABELS = {
    internal_key: human_label
    for mapping in config["data_mapping"].values()
    for human_label, internal_key in mapping.items()
}

@st.cache_data(show_spinner=False)
def fetch_recent_online(tr: str = 'last 8 hours', ar = '15 minutes') -> pd.DataFrame:
    """
    Fetches recent online data from InfluxDB for the last `minutes` minutes.
    Args:
        minutes (int): Number of minutes to fetch data for.
    Returns:
        pd.DataFrame: DataFrame containing the recent online data.
    """
    selected_measurements = list(MEASUREMENT_LABELS.keys())
    combined_df = dr.fetch_online_df(selected_measurements,
                                    tr, 
                                    ar,
                                    FREQUENCY_TO_TIMEDTA,
                                    MEASUREMENT_LABELS,
                                    FIELD_LABELS)
    return combined_df


@st.cache_data(show_spinner=False, ttl=300)
def fetch_recent_online_5min(tr: str = "last 8 hours") -> pd.DataFrame:
    """Fetch recent data cached for ~5 minutes (demo widgets)."""
    selected_measurements = list(MEASUREMENT_LABELS.keys())
    return dr.fetch_online_df(
        selected_measurements,
        tr,
        "5 minutes",
        FREQUENCY_TO_TIMEDTA,
        MEASUREMENT_LABELS,
        FIELD_LABELS,
    )

# ────────────────────────────────────────────────────────────────────────────────
# 4) PACKETS & PROMPTS
# ────────────────────────────────────────────────────────────────────────────────

def df_packet(df: pd.DataFrame, max_rows: int = 160) -> str:
    """
    Converts a DataFrame to a markdown packet for display in Streamlit.
    Args:
        df (pd.DataFrame): DataFrame to convert.
        max_rows (int): Maximum number of rows to display in the packet.
    Returns:
        str: Markdown formatted string with the DataFrame and summary stats.
    """
    if df.empty:
        return "_No data in the selected window._"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].astype(float).round(4)
    if len(d) > max_rows:
        step = max(1, len(d)//max_rows)
        d = d.iloc[::step]
    parts = []
    parts.append(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    parts.append(d.reset_index(names="timestamp").to_markdown(index=False))
    parts.append("\n**Summary Stats:**")
    parts.append(df.describe().round(3).to_markdown())
    return "\n\n".join(parts)

def build_review_prompt(target_label: str, df1: pd.DataFrame, df2: pd.DataFrame | None, best_snap: dict) -> str:
    """
    Builds a review prompt for the AI copilot based on the target label and dataframes.
    Args:
        target_label (str): The target label for the review.
        df1 (pd.DataFrame): DataFrame for the first timeframe.
        df2 (pd.DataFrame | None): DataFrame for the second timeframe (optional).
        best_snap (dict): Dictionary with the best snapshot information.
    Returns:
        str: Formatted prompt string for the AI copilot.
    """
    metric = METRIC_MAP[target_label]
    pkt1 = df_packet(df1[[metric]].dropna()) if metric in df1.columns else "_(Metric absent in CSV)_"
    pkt2 = df_packet(df2[[metric]].dropna()) if (df2 is not None and metric in df2.columns) else "_(No Timeframe 2)_"
    best_line = f"{best_snap['value']:.3f} at {best_snap['when']}" if best_snap else "N/A"
    return f"""
You are a senior blast furnace advisor. Be concise, numeric, and actionable.

# Target
- Metric: **{target_label}** (`{metric}`)
- Historical best (Apr–Jun 2024): **{best_line}** (ETA CO → max; Total Fuel → min)

# Data
## Timeframe 1
{pkt1}

## Timeframe 2
{pkt2}

# Output
0) If you sense the furnace is shutdown, reply "Furnace is shutdown, no data available."
1) Executive verdict (2–3 lines).
2) Drivers of difference (hot blast temp/vol/press, O₂, steam, PCI, top pressure, permeability, silicon, heatloads, etc.).
3) Recommendations (setpoint nudges + rationale + trade-offs).
4) Gap to historical best and what to emulate.
5) Comparison of last 2days operation vs 7days vs 30days (if available).
"""

def build_unitcost_prompt() -> str:
    return """
You are a blast furnace burden advisor. Analyze the impact of process parameters and raw material composition on **Unit Cost**.
I have priorly done this analysis and found key drivers and best practices. Now generate the findings
without hallucinating in a slightly concise manner. Donot repeat yourself and do not report mathematical analysis details (like betas,  rho). 
Report everything as Markdown only. 

# 🔍 High-Confidence Findings (Statistically Significant Drivers)

**Sign convention:**  
- Negative = higher value → lower fuel rate (fuel-saving)  
- Positive = higher value → higher fuel rate (fuel-raising)  
(Standardized OLS β shown for relative effect size; all significant with *p* < 0.05)

---

## 1️⃣ Strongest Levers During Apr–Jun 2024 (All Models Agree)

### **Hot Blast Pressure (NEGATIVE)**
- Spearman ρ ≈ −0.49; RF importance notable; OLS significant negative.  
- Apr–Jun higher by **+0.033 bar** (*p* ~ 3.0e-73).  
➡️ Higher wind pressure strongly linked to lower fuel rate.

---

### **Hot Blast Volume (NEGATIVE)**
- Negative direction in OLS and correlations; non-trivial RF importance.  
- Apr–Jun higher by **+6,543 Nm³/hr** (*p* ≪ 1e-10).  
➡️ More wind volume aligned with the low-fuel window.

---

### **Flux Addition Rate (FLUX_MT) (NEGATIVE)**
- Spearman ρ ≈ −0.27; RF ranked high; OLS coefficient negative.  
- Apr–Jun higher by **+0.57 t/h** (*p* ≈ 1.4e-5).  
➡️ More flux correlated with lower fuel at current burden chemistry.

---

### **Sinter Basicity (NEGATIVE)**
- RF and OLS show lower basicity aligns with lower fuel.  
- Apr–Jun slightly lower (−0.0019; *p* ≈ 0.20).  
➡️ Small but consistent with fuel-saving effect.

---

### **Coke Ash % (POSITIVE)**
- Higher ash increases fuel demand.  
- Apr–Jun lower by **−0.38 %** (*p* ≪ 1e-10).  
➡️ Lower coke ash supported the fuel reduction.

---

### **Na₂O in Sinter (POSITIVE)**
- Higher Na₂O raises fuel demand.  
- Apr–Jun lower by **−0.0011 %** (*p* ≪ 1e-20).  
➡️ Keeping Na₂O low is beneficial.

---

📌 **Net Effect (Apr–Jun 2024):**  
- **Fuel-saving ↑:** Wind pressure, wind volume, flux rate  
- **Fuel-saving ↓:** Coke ash, Sinter Na₂O, Sinter basicity  

---

## 2️⃣ Levers That Moved *Against* the Fuel Decrease

### **Injected Fuel (ActualKg/Thm., PCI) (POSITIVE, Strongest)**
- RF importance ~0.73; β ≈ +4.93.  
- Apr–Jun higher by **+15.8 kg/thm** (*p* ≪ 1e-10).  
➡️ PCI raised Act. Fuel Rate, but was offset by wind/material gains.

---

### **Hot Blast Temperature °C (POSITIVE)**
- Positive correlation with fuel rate.  
- Apr–Jun higher by **+13 °C** (*p* ≪ 1e-200).  
➡️ Likely operational coupling, not causal.

---

### **Sinter Al₂O₃ % (POSITIVE)**
- Higher alumina raised fuel.  
- Apr–Jun slightly higher (+0.076 %; *p* ≪ 1e-60).  
➡️ Reducing Al₂O₃ helps.

---

## ✅ Which Variables Drove the Low Fuel Rate?

**Fuel-reducing (↑ increased):**
- Hot Blast Pressure  
- Hot Blast Volume  
- Flux Rate  

**Fuel-reducing (↓ decreased):**
- Coke Ash %  
- Sinter Na₂O %  
- Sinter Basicity  

**Counter-direction (increased fuel, but offset):**
- PCI (Actual Fuel Rate)  
- Hot Blast Temp °C  
- Sinter Al₂O₃ %  

---

## 📌 Actionable Operating Guidance (Data-Driven)

- **Maintain higher HB pressure & wind volume** within safe limits.  
- **Sustain or increase flux rate** while balancing slag chemistry.  
- **Procure lower coke ash** (≤ Apr–Jun median).  
- **Keep Na₂O in sinter low** via blending & fines control.  
- **Maintain slightly lower basicity** (avoid over-basic burdens).  
- **Be cautious with HBT** – correlation is operational, not causal.  
- **PCI**: If the goal is absolute min. fuel, consider steady or reduced PCI at high wind/low-ash conditions.

---

# ⚖️ Optimization for Unit Fuel Cost

### Definition
FuelCostEq (kgCokeEq/thm)} = Coke Rate + 0.53 * PCI

---

## 🔑 Key Results
- **Apr–Jun 2024 Avg:** **483.0 kgCokeEq/thm**  
- **Best Historic 7-Day Run (B7D):** **473.4 kgCokeEq/thm**  
  (≈ **−9.6** vs Apr–Jun)  
- **Production:** A–J **89.96 t/h** vs B7D **82.51 t/h**  
  ➡️ B7D was cheaper but ran slower.

---

## 📉 What Drove Low Cost in Apr–Jun 2024

**Operational Drivers**
- HB Pressure ↑ → Cost ↓ (β ≈ −1.78, *p*≪0.001)  
- Wind Volume ↑ → Cost ↓  
- Flux Rate ↑ → Cost ↓ (β ≈ −1.08, *p*≪0.001)  
- PCI ↑ → Cost ↓ (−0.35 kgCokeEq saved per +1 kg PCI)  
- HB Temp ↑ → Cost ↑ (+0.15 per 1 °C)

**Burden Chemistry**
- Lower K₂O & Na₂O → Cost ↓  
- Lower FeO, SiO₂, TiO₂ → Cost ↓  
- Higher Al₂O₃ → Cost ↑

**Sensitivities**
- O₂ Enrichment: −2.58 kg/thm per +1 %  
- HB Pressure: −4.4 per +0.05 bar  
- Flux: −0.86 per +1 t/h  
- HB Temp: +1.53 per +10 °C  
- Wind Volume: −2.5 per +5,000 Nm³/h  

---

## 🔁 How the Best 7-Day Run Got Cheaper

**Cost-Reducing Changes (B7D vs Apr–Jun):**
- PCI +5.95 kg/thm (saved cost)  
- O₂ Enrichment +0.93 % abs.  
- Cleaner burden: Ore Al₂O₃ −1.80 %, Ore K₂O −0.019 %, Sinter K₂O −0.042 %  

**Offsets (fuel-raising moves but outweighed):**
- HB Pressure −0.18 bar, Wind −16,400 Nm³/h  
- Flux −1.36 t/h  
- HB Temp −12.4 °C (helpful)

**Trade-Off:** Lower throughput (−7.45 t/h).

---

# 📊 Operating Envelope for Low Cost

**Based on lowest-cost decile (not strict limits):**

- PCI: **135–143 kg/thm**  
- O₂ Enrichment: **2.55–3.56 %**  
- HB Pressure: **2.65–2.70 bar**  
- HB Temp: **1149–1164 °C**  
- Flux: **18.6–22.9 t/h**  
- Wind: **≥115k Nm³/h**  
- Sinter K₂O: **≤0.11 %**  
- Sinter Na₂O: **0.045–0.053 %**  
- Ore K₂O: **≤0.031 %**  
- Ore Al₂O₃: **≥3.3 % (maintain slag balance)**

📌 Practical: To minimize cost, **lean on PCI + O₂**, **keep HB pressure up**, **moderate HB temp**, **add flux**, and **suppress alkalis**.

---

# ✅ Action Checklist

1. **Target PCI** where substitution ratio {d Coke}/{d PCI} <= -0.53).  
   (You are at −0.88 to −0.91 → PCI is cost-saving).  
2. **Hold HB pressure high** (2.65–2.70 bar).  
3. **Increase O₂ enrichment** (3.0–3.5 % if feasible).  
4. **Maintain flux ≥ 19 t/h**.  
5. **Manage chemistry**: Keep K₂O/Na₂O low, blend ore to reduce alkalis.  
6. **Throughput trade-off**: B7D was cheaper but slower → optimize cost *and* t/h jointly.


"""

def build_report_prompt(df: pd.DataFrame, label: str) -> str:
    pkt = df_packet(df.dropna(axis=1, how="all"))
    return f"""
You are a blast furnace reviewer. Create a structured report for **{label}**.

# Data
{pkt}

# Deliverables
- Operations snapshot (throughput, Total Fuel, ETA CO, stability).
- Thermal profile (top/bosh temps, skin temps by level, heatloads, ΔT).
- Burden quality & quantity (coke/nutcoke, sinter, CLO; PCI quality).
- Outputs (HM/slag key analysis—e.g., silicon).
- Deviations vs typical.
- Recommendations (concrete levers + expected effect).
"""

def build_bunker_unitcost_prompt() -> str:
    return f"""

You are a blast furnace burden advisor. Analyze the impact of mainly the burden distribution on **Unit Cost** and to some extent
combined with the rawmaterial and process parameters.
I have priorly done this analysis and found key drivers and best practices. Now generate the findings
without hallucinating in a slightly concise manner. Donot repeat yourself and do not report mathematical analysis details (like betas,  rho). 
Report everything as Markdown only. 

Findings:

# What the data say (linked to your Unit Cost)

### Reminder of Unit Cost
`Unit_Cost = Coke Rate Kg/Thm + 0.53 × ActualKg/Thm.`

---

## Modeling approach
**Model:** Transparent linear model (OLS) to reduce confounding.

**Controls:**  
- Hot Blast Temp  
- TopPressureBar  
- O₂ Enrichment %  
- ETA CO *(FurnaceTopGasAnalysisCO2ETACO)*  
- PCI ActualKg/Thm

**Burden features:**  
- `portions_total_COKE`, `portions_total_NON COKE`  
- `angle_wmean_COKE`, `angle_wmean_NON COKE`  
- `outer_share_COKE`, `outer_share_NON COKE`  
- `lmg_angle`

**Model quality:**  
- **R² ≈ 0.43** on ~5,900 valid rows *(burden + controls explain ~43% of Unit Cost variance)*.  
- *(Charts: “Standardized Effects…” visualize these effects.)*

---

## Most influential & directionally consistent effects *(holding controls constant)*
- **More NON-COKE portions → lower Unit Cost** *(strong, significant).*  
  *Intuition:* better ore/sinter coverage enabling efficient gas flow.
- **More COKE portions → higher Unit Cost** *(strong, significant).*  
  *Intuition:* coke rings consume more and drive cost up.
- **Pushing NON-COKE outward (higher `angle_wmean_NON COKE`) → lower Unit Cost** *(significant).*  
  *Intuition:* spreading ore toward the periphery improves permeability/ETA and saves fuel.
- **Higher LMG angle → slight reduction in Unit Cost** *(small but significant negative coefficient in a sparse, de-collinear model).*
- **Outer share of COKE:** small negative coefficient *(cost ↓)*, not statistically strong once other features enter.

---

## Process controls behaved as expected
- **Higher PCI rate** and **higher ETA CO** both **reduce Unit Cost**.
- **Higher Hot Blast Temp** **increases** cost *(likely proxying for periods requiring more heat input).*

---

## “Best burden distributions” found in your history
Each distribution change interval was evaluated by the **realized mean Unit Cost** until the next change (and we also tracked Coke rate, PCI, and ETA CO).

**Top performing change windows (≥300 data rows; long enough to trust):**  
*(See full table in “Top 25 Best Burden Events…”)*
1. **2024-03-28 17:00 — mean Unit Cost ≈ 487.0**  
   - COKE portions: **11**, NON-COKE portions: **8**  
   - `angle_wmean_COKE` ≈ **26.0°**, `angle_wmean_NON-COKE` ≈ **28.0°**  
   - `outer_share_NON-COKE`: **0.25**  
   - **LMG angle:** **42.5** *(pattern “P TO C”)*  
   - **Purpose:** “TO IMPROVE THE CENTER GAS FLOW”.

2. **2024-11-08 12:47 — mean Unit Cost ≈ 489.5**  
   - COKE portions: **37**, NON-COKE portions: **24** *(multiple ring sets logged at the same timestamp; summed)*  
   - `angle_wmean_COKE` ≈ **27.4°**, `angle_wmean_NON-COKE` ≈ **28.5°**  
   - `outer_share_NON-COKE`: **0.25**  
   - **Purpose:** “TO INCREASE THE UTILISATION”.

3. **2025-08-02 10:00 — mean Unit Cost ≈ 492.0**  
   - COKE: **11**, NON-COKE: **8**  
   - `angle_wmean_COKE` ≈ **26.7°**, `angle_wmean_NON-COKE` ≈ **28.0°**  
   - `outer_share_NON-COKE`: **0.25**

4. **2024-08-20 20:00 — mean Unit Cost ≈ 493.9**  
   - COKE: **11**, NON-COKE: **8**  
   - `angle_wmean_COKE` ≈ **26.8°**, `angle_wmean_NON-COKE` ≈ **28.3°**  
   - `outer_share_NON-COKE`: **0.33**  
   - **Purpose:** “TO CONTROL PRE”.

---

## Common pattern across best windows
- **Moderate COKE portions (~10–11)** & **adequate NON-COKE portions (~8)**.  
- **NON-COKE weighted angle near ~28°** and **≥25% in the outer ring (≥32°)**.  
- **LMG angle ~40–43** with **P→C charging pattern** frequently noted.  
- These windows also show **good ETA CO** and **healthy PCI**.

> **Rule of thumb (from your data):**  
> Keep **coke portions lean**, keep **non-coke portions ample**, and **bias the non-coke outward** *(center-of-mass ~28° with ≥25% outer share).*

**Key drivers summary:**  
- **NON-COKE `portions_total`** is the strongest cost **reducer**.  
- **COKE `portions_total`** is the strongest cost **increaser**.  
- **NON-COKE weighted angle** reduces cost *(more outer non-coke).*

---

## Why this is faithful to your physical process
- Respects the **6-row block design**; carries **date/time forward** so every material record is stamped correctly.
- Pairs each **“RINGS”** row with its **following “Angle”** row **only** to obtain degrees *(avoids double-counting portions)*.
- Separates **`metric=portions`** vs **`metric=percent`** so Extra-Coke “IN %” entries don’t pollute portions counts.
- Models **change windows** *(pattern holds until next change)*, not isolated points.

---

## Actionable next steps (recommended)
- **Lock a candidate best pattern from history:**
  - **~10–11 COKE portions**, **~8 NON-COKE portions**
  - **Target `angle_wmean_NON-COKE` ≈ 28°**, **≥25% outer share**
  - **LMG angle ~40–43**, **charging pattern P→C**
"""

def build_anomaly_prompt(recent_df: pd.DataFrame, notes: str = "") -> str:
    # Compact alerts table
    pkt = recent_df if not recent_df.empty else "_No timeseries to show_"
    return f"""
You are an anomaly spotter. Report the key anomalies in last 8hours shift (provided data) using Z-score in one line per issue.

You also received raw data (recent_df) for:
a. blast furnace temperature profile denoted "Temperature Profile - BF2_BFBF Furnace Body [furnace_level]mm Temp [circumferential_position]"
    Description for Number of sensors at different levels and any proxy name (furnace profile) is given below
    Desc:
        "4373":
          n_sensors: 7
        "5411":
          n_sensors: 13
        "5757":
          proxy_name: "Hearth"      
          n_sensors: 13
        "6103":
          n_sensors: 13
        "6795":
          n_sensors: 12
        "7565":
          n_sensors: 14
        "8335":
          n_sensors: 14
        "9105":
          n_sensors: 12
        "12975":
          proxy_name: "Bosh"   
          n_sensors: 4
        "15162":
          proxy_name: "Belly"  
          n_sensors: 4
        "18660":
          proxy_name: "Stack"
          n_sensors: 4

        "Tuyeres" are located at 10500mm
b. heatload at different levels denoted by row number. 
    Ex: "Heatload Delta T - Heat load R8 Q3 (Stave No 17-24)" - Average heatload for Quadrant 3 (for staves 17 to 24)
        "Heatload Delta T - Heat load Row6-10 Q1 (Stave No 17-24) - Average for Rows 6 to 10 but only Quadrant 3
        "Heatload Delta T - Heat load Row6" - for average heatload in row 6 across all quadrants
c. process_params:
    - blast furnace top pressure, volume, temperature, O₂, steam, PCI (coal rate) etc

Review the **last 8 hours** for furnace profile temperature spikes, heatload spikes, ΔT excursions,
gas/pressure instabilities.

# Recent 8hours packet (averaged to 15mins)
{pkt}

# Operator notes
{notes}

# Output in brief upto 200 words only
- Key observations (2–3 lines) for operator of what happened in previous shift. 
Like Blowdowns, StartUps, Shutdowns.
Are heatloads increasing?
Is fuel rate increasing?
Is blast furnace stable?

- Alerts (issue + severity).
- Likely causes mapped to controllables (HB temp/volume/pressure, O₂, steam, PCI) and burden quality.

NOTE: 1. Avoid any hallucincations and only stick to provided data. Don't be verbose and mention each point only once. 
2. Currently the operator does not have access to provide prompt feedback. So dont ask questions/expect further input.
"""


# ────────────────────────────────────────────────────────────────────────────────
# 6) STREAMLIT UI
# ────────────────────────────────────────────────────────────────────────────────

st.title("🤖 AI Copilot")

combined_df = fetch_recent_online()
# Display and allow download
if isinstance(combined_df, pd.DataFrame) and not combined_df.empty:
    combined_df = combined_df.sort_index()
    st.dataframe(combined_df)
else:
    st.info("No data returned for the selected options. Will wait for 15 minutes to refresh.")

# === Static CSV (used by Review & Drivers) ===
df_static = load_static_df(STATIC_CSV_PATH)

st.header("Analysis & Reports")
tabs = st.tabs(["Unit cost", "Unit cost & Burden Dist", "Anomalies"])

# ── Review Tab (static CSV) ─────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Unit cost review - Apr–Jun 2024")

    if st.button("Generate Review"):

        system = "You are a precise, senior blast furnace advisor. Be concise, numeric, and actionable."
        prompt = build_unitcost_prompt()

        with st.spinner("Generating review…"):
            out = call_llm(system, prompt)
        st.markdown(out)

# ── Report Tab (static CSV) ─────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Unit cost & Burden Dist")
    if st.button("Generate Review - Burden Dist"):

        system = "You are a precise, senior blast furnace advisor. Be concise, numeric, and actionable."
        prompt = build_bunker_unitcost_prompt()

        with st.spinner("Generating review…"):
            out = call_llm(system, prompt)
        st.markdown(out)

# ── Anomalies Tab (Influx) ─────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Anomalies")
    notes = st.text_area("Operator notes (optional)")
    # minutes = st.slider("Lookback (minutes)", 5, 60, 15, step=5)

    with st.expander("Channeling propensity (demo)"):
        show_channeling = st.checkbox(
            "Compute channeling propensity (updates every 5 minutes)",
            value=False,
            key="channeling_propensity_enable",
        )

        if show_channeling:
            with st.spinner("Computing propensities…"):
                df_5m = fetch_recent_online_5min(tr="last 8 hours")
                suite = compute_propensity_suite(df_5m)

            if not suite:
                st.info("Not enough data/columns to compute propensities.")
            else:
                items = list(suite.items())
                cols = st.columns(min(4, len(items)))
                for i, (name, res) in enumerate(items[:4]):
                    cols[i].metric(
                        name,
                        f"{res.score_0_100:.0f}/100",
                        "ALARM" if res.alarm else "OK",
                    )

                selected_name = st.selectbox(
                    "Plot",
                    options=list(suite.keys()),
                    index=0,
                    key="propensity_plot_select",
                )
                selected = suite[selected_name]

                st.caption(
                    f"Last sample: {selected.last_timestamp.isoformat() if selected.last_timestamp is not None else '—'} | {selected.alarm_reason}"
                )
                if selected.series_5min.empty:
                    st.info("No series to plot for this propensity.")
                else:
                    st.line_chart(selected.series_5min, height=180)

    if st.button("Check Anomalies"):
        with st.spinner("Fetching recent data from Influx…"):
            df_recent = fetch_recent_online(tr='last 8 hours', ar='15 minutes')
        if df_recent.empty:
            st.warning("No recent data fetched from Influx. Check credentials/fields.")
        else:
            system = "You are a careful anomaly detector for blast furnace thermal and gas behavior."
            prompt = build_anomaly_prompt(df_recent, notes)
            with st.spinner("Summarizing anomalies…"):
                out = call_llm(system, prompt)
            st.markdown(out)


# ────────────────────────────────────────────────────────────────────────────────
# 7) FOOTER / TIPS
# ────────────────────────────────────────────────────────────────────────────────

with st.expander("⚙️ Setup notes & knobs"):
    st.markdown("""
- **OpenAI Responses API** with `code_interpreter` is enabled. Set `OPENAI_API_KEY` and (optionally) `OPENAI_MODEL`.
- **Static CSV**: page reads `src/data/V3_df_filtered.csv` for Review/Report/Drivers. Timestamps are auto-detected.
- **InfluxDB**: set `INFLUX_URL`, `INFLUX_ORG`, `INFLUX_TOKEN`. Bucket is hard-coded to `bf2_evonith_raw`.
- **Fields**: tweak `PROCESS_FIELDS`, `HEATLOAD_FIELDS`, `DELTA_T_FIELDS`, `COOLING_WATER_FIELDS`, and `TEMP_PROFILE_FIELDS`.
- **Anomaly thresholds**: simple z-score/Δz; adjust window lengths or add domain thresholds as needed.
    """)

# ────────────────────────────────────────────────────────────────────────────────
# 8) OPERATOR FEEDBACK
# ────────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Operator feedback")

# Initialize session state for feedback widgets
if "op_fb_vote" not in st.session_state:
    st.session_state["op_fb_vote"] = None
if "op_fb_text" not in st.session_state:
    st.session_state["op_fb_text"] = ""
if "op_feedback" not in st.session_state:
    st.session_state["op_feedback"] = []  # placeholder store; persist later

c_up, c_down = st.columns(2)
with c_up:
    if st.button("👍 Useful", key="op_fb_up"):
        st.session_state["op_fb_vote"] = "up"
        st.session_state["op_fb_text"] = ""
with c_down:
    if st.button("👎 Not useful", key="op_fb_down"):
        st.session_state["op_fb_vote"] = "down"

# Show textbox only when a thumbs-down is recorded
if st.session_state.get("op_fb_vote") == "down":
    st.text_area(
        "Operator feedback (optional)",
        key="op_fb_text",
        placeholder="What was not useful or how could this be improved?",
    )
    if st.button("Submit feedback", key="op_fb_submit"):
        st.session_state["op_feedback"].append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "vote": "down",
            "text": st.session_state.get("op_fb_text", ""),
        })
        # TODO: Persist this to a vector DB / persistent store for later grounding
        st.success("Thanks for the feedback. It will be used to improve future responses.")
        st.session_state["op_fb_vote"] = None
        st.session_state["op_fb_text"] = ""
elif st.session_state.get("op_fb_vote") == "up":
    st.info("Thanks for confirming it was useful.")

