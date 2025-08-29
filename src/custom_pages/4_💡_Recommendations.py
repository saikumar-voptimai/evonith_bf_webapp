import streamlit as st
import numpy as np
import pandas as pd
import os
import io

from openai import OpenAI

import joblib
from utils import optimiser, recommendations
from datetime import datetime
from pathlib import Path
from config.config_loader import load_config

config = load_config()
config_vsense = load_config('setting_vsense.yml')

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
    <h1 style="text-align: center; font-family: 'Times New Roman', Times, serif; color: black;">
        V-OptimAIse & Evonith Metallics Limited
    </h1>
    """,
    unsafe_allow_html=True
)

st.divider()

# Section 1: Select the optimisation type
optimisation_type = st.selectbox(
    "Select Optimisation Type",
    list(config_vsense['Optimisation'].keys())
)

outputs = [config_vsense['Optimisation'][model]['output_param'] for model in list(config_vsense['Optimisation'].keys())]

# Load the configuration and model
ip = config['Optimisation']['input_params']
ip_flat_list = [val for group in config['Optimisation']['input_params'].values() for val in group]

cp_dict_vals = config['Optimisation']['control_params']['Actual'].values()
cp_list = [list(cp_dict_vals)[i]['NameInMLData'] for i in range(len(cp_dict_vals))]

op_dict_vals = config['Optimisation']['output_params'].values()
op_list = [list(op_dict_vals)[i]['NameInMLData'] for i in range(len(op_dict_vals))]

cp_op_ml_dict, meas_list = {}, []
for i, param in enumerate(cp_list):
    cp_op_ml_dict[param] = {'InfluxBucket': list(cp_dict_vals)[i]['InfluxBucket'],
    'InfluxMeasurement': list(cp_dict_vals)[i]['InfluxMeasurement'],
    'InfluxName': list(cp_dict_vals)[i]['InfluxName']}
    meas_list.append(list(cp_dict_vals)[i]['InfluxBucket'] + '/' + list(cp_dict_vals)[i]['InfluxMeasurement'])

for i, param in enumerate(op_list):
    cp_op_ml_dict[param] = {'InfluxBucket': list(op_dict_vals)[i]['InfluxBucket'],
    'InfluxMeasurement': list(op_dict_vals)[i]['InfluxMeasurement'],
    'InfluxName': list(op_dict_vals)[i]['InfluxName']}
    meas_list.append(list(op_dict_vals)[i]['InfluxBucket'] + '/' + list(op_dict_vals)[i]['InfluxMeasurement'])
    
meas_set = set(meas_list)
live_data = recommendations.fetch_live_data(cp_op_ml_dict, meas_set)

live_data['Coke Rate Kg/Thm'] = live_data['Coke Rate Kg/Thm'] + live_data['nut_coke_rate']
live_data['ActualKg/Thm.'] = live_data['Act. Fuel RateKg/Thm.'] - live_data['Coke Rate Kg/Thm']
live_data['ProductionTonnesPerHr'] =  live_data['ProductionTonnesPerHr']/10

data_rel_path = config['DATA']
data_path = Path(__file__).resolve().parents[1] / data_rel_path.split('/')[1] / data_rel_path.split('/')[2]
df_data = pd.read_csv(data_path, index_col=0, parse_dates=True)

new_live_row = df_data.iloc[-1].copy()

update_cols = [c for c in live_data.columns if c in df_data.columns]
live_series = live_data.iloc[0][update_cols]
new_live_row.loc[update_cols] = live_series

new_df = pd.DataFrame([new_live_row.values], columns=df_data.columns, index=[pd.to_datetime(live_data.index[-1])])
df_live= pd.concat([df_data, new_df])

models = {}
for i, opt_type in enumerate(list(config_vsense['Optimisation'].keys())):
    relpath = config_vsense['Optimisation'][opt_type]['model']
    model_path = Path(__file__).resolve().parents[1] / relpath.split('/')[1] / relpath.split('/')[2]
    models[outputs[i]] = joblib.load(model_path)

# Set the target output based on the optimisation type
target_output = config_vsense['Optimisation'][optimisation_type]['output_param']

# # Section 2: Set the starting point for the model
# cols = st.columns(2)
# with cols[0]:
#     date = st.date_input("Select Date")
# with cols[1]:
#     time = st.selectbox("Select Time Index", [f"{i}:00:00" for i in range(24)])

# date_time = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
TIME_IDX = -1 # int(np.where(pd.to_datetime(df_data.index, format="%d/%m/%Y %H:%M") < date_time)[0][-1])

# st.toast(f"Using data at {df_data.index[TIME_IDX]} as event time for optimisation.")

# Section 3: Display the current data and control parameters
st.subheader("Control Parameters")

prev_params = {}
include_control = {}
cols = st.columns(3)
i = 1
with st.form(key="Control Params Form"):
    for cp in cp_list:
        with cols[(i % 3) - 1]:
            prev_default_val = df_live[cp].iloc[-1]
            user_val = st.number_input(
                cp, 
                min_value=np.min([float(df_live[cp].min()), float(prev_default_val)]), 
                max_value=np.max([float(df_live[cp].max()), float(prev_default_val)]), 
                value=float(prev_default_val)
            )
            override = st.checkbox(f"Override", key=f"ov_{cp}")
            if override:
                include_control[cp] = user_val
            else:
                include_control[cp] = np.nan
            i += 1
    input_submit = st.form_submit_button("Submit Control Params")
# User-specified input variables:
with st.expander("Input Parameters - Raw Material Data - Click to expand and override"):
    cols = st.columns(3)
    raw_mtrl_input = {}
    # Display input boxes for Flux Parameters
    keys = list(ip.keys())
    with st.form(key="Raw Material Input Form"):
        for i, (key, ip_flat) in enumerate(ip.items()):
            with cols[(i+1) % 3]:
                st.write(f"### {key} Parameters")
                for i, param in enumerate(ip[key]):
                    default_val = df_live[param].iloc[-1]
                    user_val = st.number_input(param, 
                                               format="%.2f", 
                                               min_value=df_live[param].min(), 
                                               max_value=df_live[param].max(),
                                               value=default_val)
                    if user_val != default_val:
                        raw_mtrl_input[param] = user_val
                    else:
                        raw_mtrl_input[param] = np.nan
        input_submit = st.form_submit_button("Submit Input Params")

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
# Every form must have a submit button.
if st.button("Run Optimiser"):
    fixed_cp = {cp: val for cp, val in include_control.items() if not np.isnan(val)}
    user_input = {param: raw_mtrl_input.get(param, np.nan) for param in ip_flat_list}
    df_data_processed = recommendations.process_dataframe(df_live,
                                                          target_col=target_output,
                                                          targets=op_list,
                                                          lags=config_vsense['LAGS']
                                                          )
    with st.spinner('Running the optimiser'):
        optimal_solution = optimiser.run_optimiser(
            df_data_processed,
            models, 
            user_input, 
            fixed_cp,
            cp_list,
            target_output,
            optimisation_type,
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

    if (optimal_solution[target_output] - df_live[target_output].iloc[-1]) > 0:
        st.write('Already operating at optimal level.')
    else:
        st.metric(label=target_output, 
                value=f"{optimal_solution[target_output]:.2f}", 
                delta=f"{optimal_solution[target_output] - df_live[target_output].iloc[-1]:+.2f}",
                delta_color="inverse")
    
        cols = st.columns(3)
        for i, (key, new_val) in enumerate(optimal_solution.items()):
            with cols[i % 3]:
                if key in outputs and key != target_output:
                    old_val = df_live[key].iloc[-1]
                    delta = new_val - old_val
                    st.metric(label=key, value=f"{new_val:.2f}", delta=f"{delta:+.2f}")
        
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
