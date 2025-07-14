import streamlit as st
import numpy as np
import pandas as pd
import joblib
from utils import optimiser, recommendations
from datetime import datetime
from pathlib import Path
from src.config.config_loader import load_config

config = load_config()
config_vsense = load_config('setting_vsense.yml')

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
cp_list = list(set([val for group in config['Optimisation']['control_params'].values() for val in group]))

data_rel_path = config['DATA']
data_path = Path(__file__).resolve().parents[1] / data_rel_path.split('/')[1] / data_rel_path.split('/')[2]
df_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
models = {}
for i, opt_type in enumerate(list(config_vsense['Optimisation'].keys())):
    relpath = config_vsense['Optimisation'][opt_type]['model']
    model_path = Path(__file__).resolve().parents[1] / relpath.split('/')[1] / relpath.split('/')[2]
    models[outputs[i]] = joblib.load(model_path)

# Set the target output based on the optimisation type
target_output = config_vsense['Optimisation'][optimisation_type]['output_param']

# Section 2: Set the starting point for the model
cols = st.columns(2)
with cols[0]:
    date = st.date_input("Select Date")
with cols[1]:
    time = st.selectbox("Select Time Index", [f"{i}:00:00" for i in range(24)])

date_time = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
TIME_IDX = int(np.where(pd.to_datetime(df_data.index, format="%d/%m/%Y %H:%M") < date_time)[0][-1])

st.toast(f"Using data at {df_data.index[TIME_IDX]} as event time for optimisation.")

# Section 3: Display the current data and control parameters
st.subheader("Control Parameters")

prev_params = {}
include_control = {}
cols = st.columns(3)
i = 1
with st.form(key="Control Params Form"):
    for cp in cp_list:
        with cols[(i % 3) - 1]:
            prev_default_val = df_data[cp].iloc[-1]
            user_val = st.number_input(
                cp, min_value=float(df_data[cp].quantile(0.01)), 
                max_value=float(df_data[cp].quantile(0.99)), value=float(prev_default_val)
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
                    default_val = df_data[param].iloc[-1]
                    user_val = st.number_input(param, 
                                               format="%.2f", 
                                               min_value=df_data[param].min(), 
                                               max_value=df_data[param].max(),
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
        step=0.01,
        help="Regularisation parameter for the optimisation algorithm."
    )
# Every form must have a submit button.
if st.button("Run Optimiser"):
    fixed_cp = {cp: val for cp, val in include_control.items() if not np.isnan(val)}
    user_input = {param: raw_mtrl_input.get(param, np.nan) for param in ip_flat_list}
    df_data_processed = recommendations.process_dataframe(df_data,
                                                          target_col=target_output,
                                                          targets=list(config['Optimisation']['output_params']),
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
            date_time,
            lambda_reg=lambda_reg)

    st.subheader("Optimisation Results")
    # Show metrics for each control parameter and the target output
    cols = st.columns(4)
    for i, (key, new_val) in enumerate(optimal_solution.items()):
        with cols[i % 4]:
            if key in df_data.columns and key != target_output and key not in outputs:
                old_val = df_data[key].iloc[TIME_IDX]
                delta = new_val - old_val
                st.metric(label=key, value=f"{new_val:.2f}", delta=f"{delta:+.2f}")

    st.metric(label=target_output, 
              value=f"{optimal_solution[target_output]:.2f}", 
              delta=f"{optimal_solution[target_output] - df_data[target_output].iloc[TIME_IDX]:+.2f}",
              delta_color="inverse")
    
    cols = st.columns(3)
    for i, (key, new_val) in enumerate(optimal_solution.items()):
        with cols[i % 3]:
            if key in outputs and key != target_output:
                old_val = df_data[key].iloc[TIME_IDX]
                delta = new_val - old_val
                st.metric(label=key, value=f"{new_val:.2f}", delta=f"{delta:+.2f}")
    