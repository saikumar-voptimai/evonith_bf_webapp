import streamlit as st
import pandas as pd
import logging
import seaborn as sns
import plotly.express as px
from src.config import INIT_PARAMS as IPS
from config.loader import load_config
import os
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv() 

# Our custom utilities
from utils.data_submission import (
    layout_sub_sections,
    strip_section_prefix,
    flatten_single_sample_df,
    flatten_multi_sample_df_agg,
    create_production_report_xlsx
)

# --------------------------------------------------------------------------------
# HELPER FUNCTION: Submitting data for a single date
# --------------------------------------------------------------------------------
def submit_data_for_date(
    selected_date: pd.Timestamp,
    all_section_data: dict[str, dict[str, pd.DataFrame]],
    existing_df: pd.DataFrame,
):
    """
    Merges new data for the chosen date into existing_df (st.session_state["submitted_data"]).
    If the date already exists, update only the columns that have new (non-NaN) numeric values.
    If it's a new date, append a new row.

    :param selected_date: The date chosen by the user
    :param all_section_data:
        Dict of { section_key: { sub_section_name: DataFrame } }
        from layout_sub_sections. Each sub-section is either single or multi-sample.
    :param existing_df: The current DataFrame in st.session_state["submitted_data"].
    :return:
        (updated_df, changed_cols)
        updated_df: The updated DataFrame with the new columns/values
        changed_cols: A list of columns that were updated or newly created
    """
    row_list = []
    # Flatten or aggregate each sub-section
    for sec_key, sub_dict in all_section_data.items():
        for sub_name, df in sub_dict.items():
            # Remove "a. ", "b. ", etc. from sub-section name
            clean_name = strip_section_prefix(sub_name)

            # If it's multi-sample => flatten with avg/min/max
            if df.shape[0] > 1:
                flattened = flatten_multi_sample_df_agg(df, prefix=clean_name)
            else:
                flattened = flatten_single_sample_df(df, prefix=clean_name)

            row_list.append(flattened)

    # If no data, return without changes
    if not row_list:
        return existing_df, []

    # Combine horizontally => one row (index = selected_date)
    combined_row = pd.concat(row_list, axis=1)
    combined_row.index = [selected_date]

    changed_cols = []

    # Check if this date is already in the existing DataFrame
    if selected_date in existing_df.index:
        # We are updating an existing row
        old_row = existing_df.loc[[selected_date]].copy()

        # Merge new values in. If the new value is numeric (and not empty),
        # overwrite old value if old was NaN or different.
        for col in combined_row.columns:
            new_val = combined_row.loc[selected_date, col]
            old_val = old_row.loc[selected_date, col] if col in old_row.columns else None

            # If new_val is not NaN or empty, we update
            if pd.notna(new_val) and str(new_val).strip() != "":
                if pd.isna(old_val) or old_val != new_val:
                    old_row.loc[selected_date, col] = new_val
                    changed_cols.append(col)
            else:
                # If new_val is empty or NaN, do not overwrite
                pass

        # Put updated row back into existing_df
        for col in old_row.columns:
            existing_df.loc[selected_date, col] = old_row.loc[selected_date, col]

    else:
        # Brand-new date => just append
        existing_df = pd.concat([existing_df, combined_row], axis=0)
        changed_cols = list(combined_row.columns)

    return existing_df, changed_cols

def download_production_report_button(data_dict, year):
    """
    Creates an in-memory Excel file for the production report
    and offers a download button in Streamlit.
    """
    excel_data = create_production_report_xlsx(data_dict, year)
    st.download_button(
        label="Download Production Report",
        data=excel_data,
        file_name="production_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --------------------------------------------------------------------------------
# MAIN PAGE CODE
# --------------------------------------------------------------------------------
config = load_config("setting_submission.yaml")
log = logging.getLogger("root")

# Ensure session_state data exists
if "submitted_data" not in st.session_state:
    st.session_state["submitted_data"] = pd.DataFrame()

st.title("Offline Data Submission Portal")

# Select date
selected_date = st.date_input("Select Date for Submission", value=pd.Timestamp.now())
st.markdown("---")

# Build Tabs from config
section_names = list(config["sections"].keys())
tabs = st.tabs([config["sections"][sec]["display_title"] for sec in section_names])

all_section_data = {}
for idx, sec_key in enumerate(section_names):
    with tabs[idx]:
        section_cfg = config["sections"][sec_key]
        st.write(f"**Section:** {section_cfg['display_title']}")
        sub_sections = section_cfg["sub_sections"]

        # Layout the sub-sections for user input
        sub_data_dict = layout_sub_sections(sub_sections, selected_date)
        all_section_data[sec_key] = sub_data_dict

st.markdown("---")
st.subheader("Data submission")
# --------------------------------------------------------------------------------
# SUBMIT LOGIC - Merging new data or creating a new row
# --------------------------------------------------------------------------------
if st.button("Submit", help="Submits data across all tabs"):
    updated_df, changed_cols = submit_data_for_date(
        selected_date=selected_date,
        all_section_data=all_section_data,
        existing_df=st.session_state["submitted_data"]
    )
    st.session_state["submitted_data"] = updated_df

    if changed_cols:
        st.success(f"Data for submitted/updated!")
        st.write("### Columns Changed or Filled:")
        st.write(changed_cols)
    else:
        st.warning("No changes were made. (Either all fields were empty or identical.)")

    st.write("### All Data After Submission:")
    st.dataframe(st.session_state["submitted_data"])

# --------------------------------------------------------------------------------
# VIEW ALL DATA
# --------------------------------------------------------------------------------
if st.button("View All Submitted Data"):
    st.write("### All Submitted Data:")
    if st.session_state["submitted_data"].empty:
        st.info("No data has been submitted yet.")
    else:
        st.dataframe(st.session_state["submitted_data"])

st.markdown("---")

# --------------------------------------------------------------------------------
# DOWNLOAD EXCEL DEMO (Production & placeholders for other categories)
# --------------------------------------------------------------------------------
st.subheader("Download data as excel files")
cols = st.columns(3)
with cols[0]:
    # This is just a stub example showing how you'd create a data_dict 
    # (month->section->parameters->(date->value)) from st.session_state["submitted_data"].
    # Right now st.session_state["submitted_data"] is a wide table with columns = param_names.
    # You must convert it to a monthly dictionary if you want the create_production_report_xlsx style.
    # For now, just show a dummy data_dict example, or adapt your code to build it from the DataFrame.
    dummy_data_dict = {
        "May": {
            "Production Product": {
                "Total Hot Metal Production (MT)": {
                    "2023-05-01": 123.0,
                    "2023-05-02": 140.5
                },
                "Slag Generation (Calculated) (MT)": {
                    "2023-05-01": 90.0,
                    "2023-05-02": 95.5
                }
            },
            "By-Product Generation": {
                "BF(RF) Coke Breeze -16mm (MT)": {
                    "2023-05-01": 12,
                    "2023-05-02": 13
                }
            }
        }
    }

    if st.button("Production Report", help="Download the Daily Production Report, custom template."):
        download_production_report_button(dummy_data_dict, 2023)

with cols[1]:
    if st.button("Raw Material Analysis", help="Download Raw material analysis, custom template."):
        st.write("Placeholder for raw material analysis download.")
        # You can adapt a function like `create_production_report_xlsx`
        # but with a layout for raw material analysis, then st.download_button

with cols[2]:
    if st.button("Slag/Hot Metal Chemistry", help="Download Slag & Hot Metal Chemistry"):
        st.write("Placeholder for Slag/HM Chemistry download.")
        # Similarly adapt or implement another writer function for slag/hot metal


st.title("Visualisation tool")

with st.expander("How to Use DataWalker", expanded=False):
    st.markdown("""
    **DataWalker** is a tool for interactive exploration of tabular data. You can:
    - Use the **global search** bar to find any specific term across the dataset.
    - Apply filters, sort columns, and navigate through data interactively.
    - Enable or disable columns for streamlined viewing.
    """)

#--------------------------------------------------------------------------
st.subheader("Distribution plots")
cols = st.columns(2)
df = pd.read_excel("src/data/Cleaned_Dataset_BF_hourly (2).xlsx")
with cols[0]:
    x = st.selectbox("Select X feature", df.columns[1:])

with cols[1]:
    y = st.selectbox("Select Y feature", IPS.OUTPUT_PARAMS)
fig = sns.jointplot(data=df, x=x, y=y, kind='reg')
st.pyplot(fig, use_container_width=False)


#--------------------------------------------------------------------------
st.subheader("Timeseries plot")
cols = st.columns(2)
df = pd.read_excel("src/data/Cleaned_Dataset_BF_hourly (2).xlsx")
y = []
cols_y = df.columns[1:]
with cols[0]:
    y.append(st.selectbox("Select feature 1", cols_y))
    
with cols[1]:
    remain_cols = cols_y.insert(0, 'None')
    y.append(st.selectbox("Select feature 2", remain_cols))

y = [i for i in y if i != 'None']
# Plot the data
fig = px.line(
    df,
    x=df.columns[0],
    y=y,
    hover_data={df.columns[0]: "|%B %d, %Y"},
    markers=True
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)