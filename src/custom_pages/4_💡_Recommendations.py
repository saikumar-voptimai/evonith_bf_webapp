import streamlit as st
import numpy as np
import pandas as pd
import os
import io
import yaml
import json
from openai import OpenAI

import joblib
from utils import optimiser, recommendations
from pathlib import Path
from config.config_loader import load_config
from utils.helper_functions_explorer.data_retrieval import fetch_offline_data

config = load_config()
config_vsense = load_config('setting_vsense.yml')
field_mapping = config_vsense.get("field_mapping", {})
CONFIG_PATH = Path("src/config/setting_vsense.yml")


# Load external CSS file
css_path = Path(__file__).resolve().parents[1] / "css-style" / "recommendation_style.css"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-5-mini")
USE_CODE_INTERPRETER = True 

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

    # Fallback to Chat Completions (older SDKs)
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        chat = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.2)
        if chat and chat.choices:
            return chat.choices[0].message.content or ""
    except Exception as err2:
        last_err = err2

    return f"LLM call failed: {last_err}"
    

# Section 0: Set the title page configuration
st.markdown(
    """
    <h1 style="text-align: center; font-family: 'Times New Roman', Times, serif; ">
        V-OptimAIse & Evonith Metallics Limited
    </h1>
    """,
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------------
#  Debug Mode Toggle
# --------------------------------------------------------
debug_on = st.sidebar.toggle("Debug", value=False)
new_steps = 3 if debug_on else 30


if config_vsense.get("OPTIM_STEPS") != new_steps:
    config_vsense["OPTIM_STEPS"] = new_steps
    yaml.safe_dump(config_vsense, open(CONFIG_PATH, "w"))


# Section 1: Select the optimisation type
optimisation_type = st.selectbox(
    "Select Optimisation Type",
    list(config_vsense['Optimisation'].keys())
)
# Extract input parameter groups for the selected optimisation type
input_params = config_vsense['Optimisation'][optimisation_type]['input_params']


outputs = [config_vsense['Optimisation'][model]['output_param'] for model in list(config_vsense['Optimisation'].keys())]
models_dict = config_vsense['Optimisation']
# Load the configuration and model
for model in models_dict.keys():
    ip_flat = [val for group in models_dict[model]['input_params'].values() for val in group]
    models_dict[model]['input_params_flat'] = ip_flat
    models_dict[model]['Optimised'] = False    
    relpath = models_dict[model]['model']
    model_path = Path(__file__).resolve().parents[1] / relpath.split('/')[1] / relpath.split('/')[2]
    models_dict[model]['LoadedMLModel'] = joblib.load(model_path)
    if model == optimisation_type:
        models_dict[model]['Optimised'] = True

cp_list = models_dict[optimisation_type]['control_params']    
cp_op_list = list(config['Parameter Synonyms'].keys())
cp_op_ml_dict, meas_list = {}, []
for i, param in enumerate(cp_op_list):
    cp_op_ml_dict[param] = {'InfluxBucket': config['Parameter Synonyms'][param]['InfluxBucket'],
    'InfluxMeasurement': config['Parameter Synonyms'][param]['InfluxMeasurement'],
    'InfluxName': config['Parameter Synonyms'][param]['InfluxName'],
    'NameInMLData': config['Parameter Synonyms'][param]['NameInMLData']}
    meas_list.append(config['Parameter Synonyms'][param]['InfluxBucket'] + '/' + 
                     config['Parameter Synonyms'][param]['InfluxMeasurement'])
    
meas_set = set(meas_list)
live_data = recommendations.fetch_live_data(cp_op_ml_dict, meas_set)
# st.dataframe(live_data)


# Calculated data:
live_data['UnitCost 1000Rs/Thm'] = live_data['Coke Rate Kg/Thm']  + config['Coke to PCI'] * live_data['ActualKg/Thm.']

# Historical Data:
data_rel_path = config['DATA']
data_path = Path(__file__).resolve().parents[1] / data_rel_path.split('/')[1] / data_rel_path.split('/')[2]
df_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
df_data.index = pd.to_datetime(df_data.index, format="%d-%m-%Y %H:%M", utc=True)
# Attach live:
new_live_row = df_data.iloc[-1].copy()


update_cols = [c for c in live_data.columns if c in df_data.columns]
live_series = live_data.iloc[0][update_cols]
new_live_row.loc[update_cols] = live_series


new_df = pd.DataFrame([new_live_row.values], columns=df_data.columns, index=[pd.to_datetime(live_data.index[-1])])
df_live= pd.concat([df_data, new_df])
# Set the target output based on the optimisation type
target_output = config_vsense['Optimisation'][optimisation_type]['output_param']

TIME_IDX = -1 # int(np.where(pd.to_datetime(df_data.index, format="%d/%m/%Y %H:%M") < date_time)[0][-1])

# st.toast(f"Using data at {df_data.index[TIME_IDX]} as event time for optimisation.")

# Section 3: Display the current data and control parameters
st.subheader("Control Parameters")

timestamp_utc = live_data.index[0]
timestamp_ist = timestamp_utc.tz_convert("Asia/Kolkata").tz_localize(None)
st.markdown(f"<h4>DATE-TIME (IST): {timestamp_ist}</h4>", unsafe_allow_html=True)
# --- File paths ---
bounds_file = Path("src/data/control_bounds.json")

# Ensure directory exists
bounds_file.parent.mkdir(parents=True, exist_ok=True)

# --- Load persisted bounds safely ---
if bounds_file.exists():
    with open(bounds_file, "r") as f:
        persisted_bounds = json.load(f)

# --- Load CSS ---
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

include_control = {}
if 'control_params' not in st.session_state:
    st.session_state['control_params'] = include_control
with st.form("Control Params Form"):
    cols = st.columns(3)
    i = 0
    for cp in cp_list:
        with cols[i % 3]:
            # Get machine limits
            col_min = float(df_live[cp].min())
            col_max = float(df_live[cp].max())
            latest_val = float(df_live[cp].iloc[-1])

            # Use persisted values if any
            cp_min = persisted_bounds.get(cp, {}).get("min", col_min)
            cp_max = persisted_bounds.get(cp, {}).get("max", col_max)
            # val = persisted_bounds.get(cp, {}).get("value", latest_val)
            val = latest_val
            override = persisted_bounds.get(cp, {}).get("override", False)

            col1, col2 = st.columns([0.05, 0.95])

            # Checkbox in the first column
            with col1:
                override = st.checkbox(" ", value=False, key=f"override_{cp}")

            # Heading in the second column, styled
            col2.markdown(
                f"<div class='param-title'>{cp}</div>",
                unsafe_allow_html=True
            )

            st.number_input(
                "Value",
                min_value=cp_min,
                max_value=cp_max,
                value=np.clip(val, cp_min, cp_max),
                key=f"val_{cp}",
            )

            # --- Row 2: MIN | MAX ---
            min_col, max_col = st.columns(2)
            if override:
                with min_col:
                    st.number_input(
                        "Min",
                        value=cp_min,
                        key=f"min_{cp}",
                        disabled=True  # Uneditable when override is checked
                    )
                with max_col:
                    st.number_input(
                        "Max",
                        value=cp_max,
                        key=f"max_{cp}",
                        disabled=True  # Uneditable when override is checked
                    )
            else:
                with min_col:
                    min_val = st.number_input(
                        "Min",
                        value=cp_min,
                        key=f"min_{cp}",
                        disabled=False  # Editable when override is unchecked
                    )
                with max_col:
                    max_val = st.number_input(
                        "Max",
                        value=cp_max,
                        key=f"max_{cp}",
                        disabled=False  # Editable when override is unchecked
                    )

            # Ensure current value is within limits
            val = min(max(val, min_val), max_val)
            include_control[cp] = {"min": min_val, "max": max_val, "value": val, "override": override}

        i += 1
    st.session_state['control_params'] = include_control
    submit_cp = st.form_submit_button("Submit CP & Save bounds")
    if submit_cp:
        st.session_state['control_params'] = include_control
        with open(bounds_file, "w") as f:
            json.dump(include_control, f, indent=4)
        st.success("✅ Bounds saved successfully!")


# User-specified input variables:
with st.expander("Input Parameters - Raw Material Data - Click to expand and override"):
    

    ml_cfg = config_vsense.get("influxdb_ml_database", {})

    # Fetch raw material input data from InfluxDB
    df_live_ip = fetch_offline_data(
        measurement=ml_cfg.get("measurements", "rm_charge_data"),
        time_range=ml_cfg.get("time_range", "last 1 month"),
        database=ml_cfg.get("database", "ML DATASET"),
    )

    df_live_ip = df_live_ip.rename(columns=field_mapping)
    def get_shift(ts):
        h = ts.hour
        return "C" if h < 8 else "A" if h < 16 else "B"


    latest_row = df_live_ip.iloc[-2]
    # st.dataframe(latest_row)
    timestamp_utc = latest_row.name
    timestamp_ist = timestamp_utc.tz_convert("Asia/Kolkata").tz_localize(None)

    shift = get_shift(timestamp_ist)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<h4>DATE-TIME (IST): {timestamp_ist}</h4>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<h4>Shift: {shift}</h4>", unsafe_allow_html=True)



    cols = st.columns(3)
    raw_mtrl_input = {}

    with st.form(key="Raw Material Input Form"):

        for i, (group_name, params) in enumerate(input_params.items()):
            with cols[i % 3]:
                st.write(f"### {group_name}")
                for param in params:
                    if param in latest_row:
                        default_val = float(latest_row[param])
                    else:
                        default_val = 0.0
                    user_val = st.number_input(
                        param,
                        format="%.2f",
                        value=default_val,
                    )
                    raw_mtrl_input[param] = (
                        user_val if user_val != default_val else np.nan
                    )

        raw_mtrl_input[param] = np.nan
        submit_ip = st.form_submit_button("Submit Input Params")
        if submit_ip:
            st.success("✅ Input parameters recorded.")
ip_flat_list = [val for group in models_dict[optimisation_type]['input_params'].values() for val in group]
op_list = [config_vsense['Optimisation'][model]['output_param'] for model in list(config_vsense['Optimisation'].keys())]

with st.form("Optimiser Form"):
    cols = st.columns(2)
    with cols[0]:
        lambda_reg = st.slider(
            "Regularisation Parameter (Lambda)",
            min_value=0.0, 
            max_value=0.5, 
            value=config_vsense['LAMBDA_REG'],
            step=0.05,
            help="Regularisation parameter for the optimisation algorithm."
        )
    submit_optim_params = st.form_submit_button("Run Optimiser")
    if submit_optim_params:
        st.success("✅ Optimiser run requested.")
        fixed_cp = {cp: cp_props['value'] for cp, cp_props in include_control.items() if not np.isnan(cp_props['value']) and cp_props['override']}
        user_input = {param: raw_mtrl_input.get(param, np.nan) for param in ip_flat_list}
        df_data_processed = recommendations.process_dataframe(df_live,
                                                            target_col=target_output,
                                                            targets=op_list,
                                                            lags=config_vsense['LAGS']
                                                            )
        with st.spinner('Running the optimiser'):
            optimal_solution = optimiser.run_optimiser(
                df_data_processed,
                models_dict, 
                user_input, 
                fixed_cp,
                lambda_reg=lambda_reg)

        st.subheader("Optimisation Results")
        # Show metrics for each control parameter and the target output
        not_needed_list = ['TuyereVelocitym/s', 'Total OxygenNm3/Hr.', 'TopPressureBar']
        cols = st.columns(4)
        for i, (key, new_val) in enumerate(optimal_solution.items()):
            if key in df_live.columns and key != target_output and key not in outputs and key not in not_needed_list:
                with cols[i % 4]:
                    old_val = df_live[key].iloc[-1]
                    delta = new_val - old_val
                    if abs(delta)/abs(old_val) >= 0.01:  # Only show significant changes
                        st.metric(label=key, value=f"{new_val:.2f}", delta=f"{delta:+.2f}")
                    if abs(delta)/abs(old_val) < 0.01:  # Only show significant changes
                        st.metric(label=key, value=f"{old_val:.2f}", delta=f"{0:+.2f}")

        st.metric(label=target_output, 
                value=f"{optimal_solution[target_output + '_current']:.2f}", 
                delta=f"{optimal_solution[target_output + '_current'] - optimal_solution[target_output + '_previous']:+.2f}")

        cols = st.columns(3)
        for i, (key, new_val) in enumerate(optimal_solution.items()):
            with cols[i % 3]:
                key_feat = key.replace('_current', '').replace('_previous', '')
                if key_feat in outputs and key_feat != target_output and '_previous' in key:
                    old_val = optimal_solution[key_feat + '_previous']
                    new_val = optimal_solution[key_feat + '_current']
                    delta = new_val - old_val
                    st.metric(label=key_feat, value=f"{new_val:.2f}", delta=f"{delta:+.2f}")
        
        # Section 4: Generate recommendations using LLM
        st.subheader("Recommendations")
        system = "You are a precise, senior blast furnace advisor. Be concise, numeric, and actionable."

        prompt = f"""

        You are a blast furnace burden advisor. Analyze the impact of process parameters and raw material composition on **Unit Cost**.
        I have priorly done this analysis and found key drivers and best practices. Now generate the findings
        without hallucinating in a slightly concise manner. Donot repeat yourself and do not report mathematical analysis details (like betas,  rho). 
        Report everything as Markdown only. 

        Based on the optimisation results, provide specific recommendations to improve {target_output}.
        - Optimal solution computed using current methodology {optimal_solution}.
        - Current operating point {new_df.to_dict()}.
        - List the top 3-5 control parameters to adjust, with their new values. Note that we say 
        the furnace is already optimally operated if the target output change is less than 1%.

        - Provide a brief rationale for each recommendation.
        - Use bullet points for clarity.
        - Avoid vague statements; be specific and data-driven.

        Previous data analysis observations only for your reference (do not repeat in output):
        

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
    Fuel_CostEq (kgCokeEq/thm) = Coke Rate + 0.53 * PCI
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

    1. **Target PCI** where substitution ratio (Coke/PCI = 0.53).  
    (You are at −0.88 to −0.91 → PCI is cost-saving).  
    2. **Hold HB pressure high** (2.65–2.70 bar).  
    3. **Increase O₂ enrichment** (3.0–3.5 % if feasible).  
    4. **Maintain flux ≥ 19 t/h**.  
    5. **Manage chemistry**: Keep K₂O/Na₂O low, blend ore to reduce alkalis.  
    6. **Throughput trade-off**: B7D was cheaper but slower → optimize cost *and* t/h jointly.
    """
        with st.spinner("Generating review…"):
            out = call_llm(system, prompt)
        st.markdown(out)
