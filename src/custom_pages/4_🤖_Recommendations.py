import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import INIT_PARAMS as IPS
from typing import List, Dict
from src.utils import optimiser

units_dict = {'UTILITY _Steam (Blast Furnace)_Tons': 'Tonnes', 'UTILITY _Steam (Turbo Blower)_Tons': 'Tonnes',
                'CBV From BlowerNm3/Hr.': 'Nm3/hr', 'Hot Blast VolumeNm3/Hr.': 'Nm3/hr',
                'ActualKg/Thm.': 'kg/thm', 'Hot Blast PressureBar' : 'Bar', 'Hot Blast Temp.oC': 'oC', 'Oxygen\nFlowNm3/Hr.': 'Nm3/Hr', 
                'SteamKgs/Hr.': 'kgs/hr', 'Tuyere\nVelocitym/s' : ' m/s', 'RAFToC' : 'oC',
                'O2 Enrichment %' : '%'}

def run_optimiser():
    """
    Run the optimiser with the provided parameters.
    """
    include_control = st.session_state['include_control']
    with st.status("Searching for optimal solutions..."):
        df_disp, prev_y_val = optimiser.run_optimiser(data_path, model_path, user_input, prev_params, control_params_list,
                                          include_control, num)

        # Display optimal control parameters and efficiency
        st.write("### Optimal Control Parameters:")
        st.dataframe(df_disp.style.format('{:.3f}'), width=1200)

    display_optimization_results(df_disp, prev_params, prev_y_val, control_params_list, units_dict)
    st.success("Optimal solutions found")

def display_optimization_results(
    optimal_df: pd.DataFrame, 
    prev_params: dict, 
    prev_y_val: float,
    control_params_list: list, 
    units_dict: dict
):
    """
    Displays the recommended control parameters (from the first row of 'optimal_df')
    as Streamlit metrics, along with the delta from 'prev_params'.
    
    :param optimal_df: DataFrame with the first row containing optimal control parameters
    :param prev_params: Dictionary of previous day’s parameters (e.g., prev_params[param_name])
    :param control_params_list: List of control parameter names (column names in optimal_df)
    :param units_dict: Dictionary mapping parameter names to their units
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
                    param_unit = units_dict.get(param, "")
                    
                    # Create a label that includes the parameter name + unit
                    label = f"{param} ({param_unit})" if param_unit else ""
                    # st.metric automatically formats arrows (green up / red down) based on sign
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

# INITIALISATION:
df_bf = pd.read_pickle(IPS.DATA)
df = pd.read_pickle(IPS.DATA)
control_params_list = IPS.CONTROL_PARAMS
num = IPS.OPTIM_STEPS
data_path = IPS.DATA
model_path = IPS.MODEL

flux_params = IPS.FLUX_PARAMS
sinter_params = IPS.SINTER_PARAMS
pellet_params = IPS.PELLET_PARAMS
coke_params = IPS.COKE_PARAMS
other_params = IPS.OTHER_PARAMS
ore_params = IPS.ORE_PARAMS

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
    # Forms submitted:
    st.sidebar.success("Control Parameters submitted")
    include_control = {}
    for i in range(len(control_params_list)):
        if is_include_control[control_params_list[i]]:
            st.sidebar.write(prev_params[control_params_list[i]])
            include_control[control_params_list[i]] = prev_params[control_params_list[i]]
        else:
            include_control[control_params_list[i]] = np.nan
    st.session_state['include_control'] = include_control

# Previous control parameters:
with st.sidebar.form("Control Params"):
    prev_params, is_include_control = {}, {}
    for i in range(len(control_params_list)):
        cola, colb = st.sidebar.columns([10, 100])
        with cola:
            prev_params[control_params_list[i]] = st.sidebar.slider(control_params_list[i],
                                                                min_value=df[control_params_list[i]].quantile(0.25),
                                                                max_value=df[control_params_list[i]].quantile(0.75),
                                                                value=df[control_params_list[i]].mean(),
                                                                key=f"slider_{control_params_list[i]}")
        with colb:
            is_include_control[control_params_list[i]] = st.sidebar.checkbox("Override",
                                                        value=False,
                                                        key=f"checkbox_{control_params_list[i]}")
    control_submitted = st.form_submit_button("Control Params", on_click=trigger)

# User-specified input variables:
# Define column layout for horizontal sections
with st.expander("Input Parameters - From Offline Data - Click to expand and override"):
    col1, col2, col3 = st.columns(3)
    user_input = {}
    # Display input boxes for Flux Parameters
    with st.form(key="Input Params"):
        with col1:
            st.write("### Flux Parameters")
            for i, param in enumerate(flux_params):
                st.write(f"{param}:")
                user_input[param] = st.number_input("", format="%.2f",
                                                    min_value=df[param].min(),
                                                    max_value=df[param].max(),
                                                    value=df[param].mean())

        # Display input boxes for Ore Parameters
        with col2:
            st.write("### Ore Parameters")
            for i, param in enumerate(ore_params):
                st.write(f"{param}:")
                user_input[param] = st.number_input("", format="%.2f",
                                                    min_value=df[param].min(),
                                                    max_value=df[param].max(),
                                                    value=df[param].mean())

        # Display input boxes for Sinter Parameters
        with col3:
            st.write("### Sinter Parameters")
            for i, param in enumerate(sinter_params):
                st.write(f"{param}:")
                user_input[param] = st.number_input("", format="%.2f",
                                                    min_value=df[param].min(),
                                                    max_value=df[param].max(),
                                                    value=df[param].mean())
        
        # Define column layout for horizontal sections
        col4, col5, col6 = st.columns(3)
        with col4:
            st.write("### Pellet Parameters")
            for i, param in enumerate(pellet_params):
                st.write(f"{param}:")
                user_input[param] = st.number_input("", format="%.2f",
                                                    min_value=df[param].min(),
                                                    max_value=df[param].max(),
                                                    value=df[param].mean())

        # Display input boxes for Ore Parameters
        with col5:
            st.write("### Coke Parameters")
            for i, param in enumerate(coke_params):
                st.write(f"{param}:")
                user_input[param] = st.number_input("", format="%.2f",
                                                    min_value=df[param].min(),
                                                    max_value=df[param].max(),
                                                    value=df[param].mean())

        # Display input boxes for Sinter Parameters
        with col6:
            st.write("### Other Parameters")
            for i, param in enumerate(other_params):
                st.write(f"{param}:")
                user_input[param] = st.number_input("", format="%.2f",
                                                    min_value=df[param].min(),
                                                    max_value=df[param].max(),
                                                    value=df[param].mean())
        # Every form must have a submit button.
        input_submit = st.form_submit_button("Submit Input Params", on_click=trigger)

# Every form must have a submit button.
if st.button("Run Optimiser", key="optimiser", on_click=trigger):
    include_control = st.session_state['include_control']
    with st.status("Searching for optimal solutions..."):
        df_disp, prev_y_val = optimiser.run_optimiser(data_path, model_path, user_input, prev_params, control_params_list,
                                        include_control, num)

        # Display optimal control parameters and efficiency
        st.write("### Optimal Control Parameters:")
        st.dataframe(df_disp.style.format('{:.3f}'), width=1200)

    display_optimization_results(df_disp, prev_params, prev_y_val, control_params_list, units_dict)
    st.success("Optimal solutions found")

# # Plot control parameters vs. efficiency
# st.write("### Control Parameters vs. Efficiency:")
# # Replace this with your code to create the plot

# Create your control parameters vs. efficiency plots
# # Create a sidebar for variable selection
# variable_selector = st.sidebar.selectbox("Select Variable", ["Variable 1", "Variable 2", "Variable 3"])
#
# # Sample data (replace with your own data)
# data = df_bf
#
# # Create two plots side by side
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#
# # Plot 1
# ax1.scatter(data[variable_selector], data["Efficiency"])
# ax1.set_xlabel(variable_selector)
# ax1.set_ylabel("Efficiency")
# ax1.set_title("Plot 1")
#
# # Plot 2
# ax2.scatter(data[variable_selector], data["Efficiency"], color="orange")
# ax2.set_xlabel(variable_selector)
# ax2.set_ylabel("Efficiency")
# ax2.set_title("Plot 2")

# Display the plots in Streamlit
# st.pyplot(fig)

# Create a DataFrame with similar combinations
# similar_combinations_df = pd.DataFrame(...)
# st.write(similar_combinations_df)
