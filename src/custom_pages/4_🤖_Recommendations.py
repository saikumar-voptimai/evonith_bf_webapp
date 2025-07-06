import streamlit as st
import numpy as np
import pandas as pd
import joblib
from typing import List, Dict
from utils import optimiser
from pathlib import Path
from src.config.config_loader import load_config

config = load_config()
config_vsense = load_config('setting_vsense.yml')

def run_optimiser(df_data: pd.DataFrame,
                  model: joblib,
                  user_input: Dict,
                  prev_params: Dict,
                  control_params_list: List[str],
                  no_of_steps: int):
    """
    Run the optimiser with the provided parameters.
    """
    include_control = st.session_state['include_control']
    with st.status("Searching for optimal solutions..."):
        df_disp, prev_y_val = optimiser.run_optimiser(df_data, 
                                                      model, 
                                                      user_input, 
                                                      prev_params, 
                                                      control_params_list,
                                                      include_control, 
                                                      no_of_steps)

        # Display optimal control parameters and efficiency
        st.write("### Optimal Control Parameters:")
        st.dataframe(df_disp.style.format('{:.3f}'), width=1200)

    display_optimization_results(df_disp, prev_params, prev_y_val, control_params_list)
    st.success("Optimal solutions found")

def display_optimization_results(
    optimal_df: pd.DataFrame, 
    prev_params: dict, 
    prev_y_val: float,
    control_params_list: list
):
    """
    Displays the recommended control parameters (from the first row of 'optimal_df')
    as Streamlit metrics, along with the delta from 'prev_params'.
    
    :param optimal_df: DataFrame with the first row containing optimal control parameters
    :param prev_params: Dictionary of previous day’s parameters (e.g., prev_params[param_name])
    :param control_params_list: List of control parameter names (column names in optimal_df)
    :param prev_y_val: Previous day's efficiency value for comparison
    """
    
    st.subheader("Recommended Control Parameters")
    st.write("Below you can see how each optimized parameter compares to yesterday's values.")

    # We assume the first row contains the optimal parameter values
    optimal_values = optimal_df.iloc[0]

    # Chunk parameters in sets of 3 or 4 to show them in multiple columns
    num_cols = 3  # Adjust as needed
    for i in range(0, len(control_params_list), num_cols):
        cols = st.columns(num_cols)
        
        # Display each parameter in the row's respective column
        for j, param in enumerate(control_params_list[i : i + num_cols]):
            with cols[j]:
                # Get recommended (optimal) value and previous day's value
                recommended_val = optimal_values[param]
                previous_val = prev_params.get(param, 0.0)
                
                # Calculate delta
                delta_val = recommended_val - previous_val
                if abs(delta_val) >= 0.01:
                    # Fetch unit from dictionary
                    label = f"{param}"
                    if not label.strip():
                        label = "Parameter"
                    st.metric(
                        label=label,
                        value=round(recommended_val, 2), 
                        delta=round(delta_val, 2)
                    )

    curr_val = np.round(optimal_df['Efficiency'][0], 2)
    delta_val = curr_val - prev_y_val
    curr_val = f"{curr_val:.2f}"
    delta_val = f"{delta_val:.2f}"
    st.metric(label='Efficiency', value=curr_val, delta=delta_val)

optimisation_type = 'coke_rate_opt'

# Load the configuration and model
model = joblib.load_model(config_vsense['MODELS'][optimisation_type])
ip = config[optimisation_type]['input_params']
ip_flat_list = [val for group in config[optimisation_type]['input_params'].values() for val in group]
cp_list = list(set([val for group in config[optimisation_type]['control_params'].values() for val in group]))

data_rel_path = config['DATA']
fullpath = Path(__file__).resolve().parents[1] / data_rel_path.split('/')[1] / data_rel_path.split('/')[2]
df_data = pd.read_csv(fullpath, index_col=0, parse_dates=True)

model_rel_path = config_vsense['MODELS'][optimisation_type]
model_path = Path(__file__).resolve().parents[1] / model_rel_path.split('/')[1] / model_rel_path.split('/')[2]
model = joblib.load(model_path)

# Create a Streamlit web app
if 'stage' not in st.session_state:
    st.session_state.stage = 0

# Define the custom CSS style for the title
# Set the title as a centered Markdown text
st.markdown(
    """
    <h1 style="text-align: center; font-family: 'Times New Roman', Times, serif; color: black;">
        V-OptimAIse & Evonith Metallics Limited
    </h1>
    """,
    unsafe_allow_html=True
)

st.divider()

# Add sliders for input parameters
st.sidebar.title("Previous Control Parameters")

if 'include_control' not in st.session_state:
    st.session_state['include_control'] = 0

def trigger():
    """
    Trigger function to handle the submission of control parameters.
    This function is called when the control parameters form is submitted.
    It updates the session state with the selected control parameters and their values.
    """
    st.success("Control Parameters submitted")
    include_control = {}
    for i in range(len(cp_list)):
        if is_include_control[cp_list[i]]:
            st.sidebar.write(prev_params[cp_list[i]])
            include_control[cp_list] = prev_params[cp_list[i]]
        else:
            include_control[cp_list[i]] = np.nan
    st.session_state['include_control'] = include_control

# Previous control parameters:
with st.form("Control Params"):
    prev_params, is_include_control = {}, {}
    for i in range(len(cp_list)):
        cola, colb = st.sidebar.columns([10, 100])
        with cola:
            prev_params[cp_list[i]] = st.number_input(cp_list[i],
                                                      min_value=df_data[cp_list[i]].quantile(0.05),
                                                      max_value=df_data[cp_list[i]].quantile(0.95),
                                                      value=df_data[cp_list[i]].mean(),
                                                      key=f"num_input_{cp_list[i]}")
        with colb:
            is_include_control[cp_list[i]] = st.checkbox("Override",
                                                        value=False,
                                                        key=f"checkbox_{cp_list[i]}")
    control_submitted = st.form_submit_button("Control Params", on_click=trigger)

# User-specified input variables:
# Define column layout for horizontal sections
with st.expander("Input Parameters - Raw Material Data - Click to expand and override"):
    cols = st.columns(len(ip.keys()))
    raw_mtrl_input = {}
    # Display input boxes for Flux Parameters
    keys = list(ip.keys())
    with st.form(key=keys[i]):
        for i, (key, ip_flat) in enumerate(ip.items()):
            with cols[i]:
                st.write(f"### {key} Parameters")
                for i, param in enumerate(ip[key]):
                    st.write(f"{param}:")
                    raw_mtrl_input[param] = st.number_input("", format="%.2f",
                                                        min_value=df_data[param].min(),
                                                        max_value=df_data[param].max(),
                                                        value=df_data[param].iloc[-1])

    # Every form must have a submit button.
    input_submit = st.form_submit_button("Submit Input Params", on_click=trigger)

# Every form must have a submit button.
if st.button("Run Optimiser", key="optimiser", on_click=trigger):
    include_control = st.session_state['include_control']
    with st.status("Searching for optimal solutions..."):
        df_disp, prev_y_val = optimiser.run_optimiser(df_data, 
                                                      model, 
                                                      user_input, 
                                                      prev_params, 
                                                      cp_list,
                                                      include_control, 
                                                      no_of_steps=config_vsense['OPTIM_STEPS'])

        # Display optimal control parameters and efficiency
        st.write("### Optimal Control Parameters:")
        st.dataframe(df_disp.style.format('{:.3f}'), width=1200)

    display_optimization_results(df_disp, prev_params, prev_y_val, cp_list)
    st.success("Optimal solutions found")
