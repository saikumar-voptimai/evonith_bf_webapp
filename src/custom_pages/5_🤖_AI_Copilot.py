# pages/ai_copilot.py
import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import uuid

from utils.helper_functions_AD_system.vector_store import store_embedding, fetch_records_by_date_shift, infer_shift
from utils.helper_functions_AD_system.data_summarizer import summarize_report, df_summary
from utils.helper_functions_explorer import data_retrieval as dr
from utils.prompts import build_unitcost_prompt, build_bunker_unitcost_prompt, build_anomaly_prompt, build_df_summary_prompt
from config.config_loader import load_config
from utils.LLMs import gpt_llm, openRouter_llm
from utils.helper_functions_AD_system.Anomaly_shift_controller import get_last_report_time, can_generate_new_report, display_wait_message

config = load_config("setting_ds_dv.yml")  # Load the configuration file
config_vsense = load_config('setting_vsense.yml')
config_map = load_config("mappings.yml")



# Local CSV for static fuel/ETA analyses
STATIC_CSV_PATH = config['DATA']

model_keys = list(config_vsense['Optimisation'].keys())
CONTROL_COLUMNS = config_vsense['Optimisation'][model_keys[1]]['control_params']
INPUT_COLUMNS = [item for sublist in config_vsense['Optimisation'][model_keys[1]]['input_params'].values() for item in sublist]
OUTPUT_COLUMNS = [config_vsense['Optimisation'][key]['output_param'] for key in config_vsense['Optimisation']]


ALL_COLUMNS = CONTROL_COLUMNS + INPUT_COLUMNS + OUTPUT_COLUMNS
BEST_START = datetime(2024, 4, 1, tzinfo=timezone.utc)
BEST_END   = datetime(2024, 6, 30, 23, 59, 59, tzinfo=timezone.utc)

    
# ────────────────────────────────────────────────────────────────────────────────
# 2) CSV LOADER (STATIC ANALYSIS: Review / Drivers)
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_static_df(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        st.warning(f"CSV not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df['datetime'], utc=True, errors="coerce")
    df.drop(columns=['datetime'], inplace=True, errors="ignore")
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
    selected_measurements = list(config_map["MEASUREMENT_LABELS"].keys())
    combined_df = dr.fetch_online_df(selected_measurements,
                                    tr, 
                                    ar,
                                    config_map["FREQUENCY_TO_TIMEDTA"],
                                    config_map["MEASUREMENT_LABELS"],
                                    FIELD_LABELS)
    return combined_df

# ────────────────────────────────────────────────────────────────────────────────
# 4) STREAMLIT UI
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
            out = gpt_llm(system, prompt)
        st.markdown(out)

# ── Report Tab (static CSV) ─────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Unit cost & Burden Dist")
    if st.button("Generate Review - Burden Dist"):

        system = "You are a precise, senior blast furnace advisor. Be concise, numeric, and actionable."
        prompt = build_bunker_unitcost_prompt()

        with st.spinner("Generating review…"):
            out = gpt_llm(system, prompt)
        st.markdown(out)

# ── Anomalies Tab (Influx) ─────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Anomalies")
    notes = st.text_area("Operator notes (optional)")

    # Generate a New Anomaly Report
    if st.button("Check Anomalies"):
        last_report_time = get_last_report_time("EBF_anomaly_summary")

        # Rate-limit logic 
        if last_report_time:
            can_generate, remaining = can_generate_new_report(last_report_time)
            if not can_generate:
                last_time_fmt = last_report_time.strftime("%d %b %Y, %I:%M %p")
                st.warning(f"🕒 Last anomaly report generated at **{last_time_fmt}**.")
                display_wait_message(remaining)
                st.stop()
        else:
            st.info("ℹ️ No previous anomaly report found — generating the first report now.")

        # Fetch latest data 
        with st.spinner("Fetching recent data from Influx…"):
            df_recent = fetch_recent_online(tr="last 8 hours", ar="15 minutes")
            if df_recent.empty:
                st.warning("⚠️ No recent data fetched from Influx. Check credentials or field names.")
                st.stop()
                
        with st.spinner("Generating anomaly report..."):
            # Step 1: summarize data
            df_summary_result = df_summary(df_recent)
            df_prompt = build_df_summary_prompt(df_summary_result)
            final_df_summary = openRouter_llm(
                "You are a helpful assistant that summarizes dataframes.", df_prompt
            )

            timestamp = datetime.now(ZoneInfo("Asia/Kolkata"))
            shift = infer_shift(timestamp)
            uid = str(uuid.uuid4())[:8]

            # store dataframe summary
            df_metadata = {
                "id": f"df_summary-{uid}-{timestamp.strftime('%Y-%m-%dT%H-%M-%S')}",
                "namespace": "EBF_df_summary",
                "timestamp": timestamp.strftime('%Y-%m-%dT%H-%M-%S'),
                "source": "blast_furnace",
                "shift": shift,
                "date": timestamp.strftime("%Y-%m-%d"),
            }
            store_embedding(final_df_summary, metadata=df_metadata)

            # run anomaly detection
            system = "You are a careful anomaly detector for blast furnace thermal and gas behavior."
            prompt = build_anomaly_prompt(df_recent, notes)

            with st.spinner("Summarizing anomalies…"):
                anomaly_report = openRouter_llm(system, prompt)
                summarized = summarize_report(anomaly_report)

                anomaly_metadata = {
                    "id": f"anomaly_summary-{uid}-{timestamp.strftime('%Y-%m-%dT%H-%M-%S')}",
                    "namespace": "EBF_anomaly_summary",
                    "timestamp": timestamp.strftime('%Y-%m-%dT%H-%M-%S'),
                    "source": "blast_furnace",
                    "shift": shift,
                    "date": timestamp.strftime("%Y-%m-%d"),
                }
                store_embedding(summarized, metadata=anomaly_metadata)

                # st.success(f"Anomaly report generated for **{condition}** condition.")
                st.text(anomaly_report)

    # Fetch Reports by Date & Shift 
    st.sidebar.markdown("### 🔍 Fetch Reports by Date & Shift")

    selected_date = st.sidebar.date_input("Select date", datetime.now().date())

    shift_options = {
        "Shift A": {"start": "00:00", "end": "08:00"},
        "Shift B": {"start": "08:00", "end": "16:00"},
        "Shift C": {"start": "16:00", "end": "23:59"},
    }
    selected_shift = st.sidebar.selectbox("Select shift", list(shift_options.keys()))

    if st.sidebar.button("Fetch Reports"):
        date_str = selected_date.strftime("%Y-%m-%d")
        shift_letter = selected_shift[-1]  

        with st.spinner(f"Fetching reports for {date_str} - Shift {shift_letter}..."):
            try:
                grouped_reports = fetch_records_by_date_shift(date_str, shift_letter)
                if not grouped_reports:
                    st.warning("⚠️ No reports found for the selected date and shift.")
                else:
                    order = [
                        ("EBF_anomaly_summary", "🚨 Anomaly Summary"),
                        ("EBF_df_summary", "📗 DataFrame Summary"),
                        ("EBF_operator_feedback", "👍 Operator Feedback"),
                        ("EBF_operator_notes", "📝 Operator Notes"),
                    ]
                    for ns, title in order:
                        key = f"{ns}_shift_{shift_letter}"
                        if key in grouped_reports:
                            r = grouped_reports[key]
                            st.subheader(title)
                            st.markdown(
                                f"<div style='padding:15px; border-radius:10px'>"
                                f"<pre style='white-space:pre-wrap; word-wrap:break-word'>{r['text']}</pre>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    st.success("✅ Reports loaded successfully for the selected date and shift.")
            except Exception as e:
                st.error(f"❌ Error while fetching reports: {e}")



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

# Feedback Buttons
st.subheader("Operator Feedback")

# Initialize defaults
if "op_feedback_vote" not in st.session_state:
    st.session_state["op_feedback_vote"] = None
if "op_feedback_text" not in st.session_state:
    st.session_state["op_feedback_text"] = ""
if "operator_feedback" not in st.session_state:
    st.session_state["operator_feedback"] = ""
if "operator_notes" not in st.session_state:
    st.session_state["operator_notes"] = ""

# ---------------------------------
# Feedback Buttons
# ---------------------------------
col_up, col_down = st.columns(2)
with col_up:
    if st.button("👍 Useful", key="fb_up"):
        st.session_state["op_feedback_vote"] = "like"
        st.session_state["op_feedback_text"] = ""
with col_down:
    if st.button("👎 Not Useful", key="fb_down"):
        st.session_state["op_feedback_vote"] = "dislike"

# If operator clicked "Not Useful"
if st.session_state.get("op_feedback_vote") == "dislike":
    feedback_text = st.text_area(
        "Optional Feedback / Reason",
        key="fb_text",
        placeholder="What was not useful or how could this be improved?",
    )

    if st.button("Submit Feedback", key="fb_submit"):
        feedback_text = feedback_text.strip()
        if not feedback_text:
            st.warning("Please add some notes before submitting.")
        else:

            timestamp = datetime.now(ZoneInfo("Asia/Kolkata"))
            shift = infer_shift(timestamp)
            uid = str(uuid.uuid4())[:8]

            feedback_metadata = {
                "id": f"operator_feedback-{uid}-{timestamp.strftime('%Y-%m-%dT%H-%M-%S')}",
                "namespace": "EBF_operator_feedback",
                "timestamp": timestamp.strftime('%Y-%m-%dT%H-%M-%S'),
                "source": "blast_furnace",
                "shift": shift,
                "date": timestamp.strftime("%Y-%m-%d"),
            }

            notes_metadata = {
                "id": f"operator_notes-{uid}-{timestamp.strftime('%Y-%m-%dT%H-%M-%S')}",
                "namespace": "EBF_operator_notes",
                "timestamp": timestamp.strftime('%Y-%m-%dT%H-%M-%S'),
                "source": "blast_furnace",
                "shift": shift,
                "date": timestamp.strftime("%Y-%m-%d"),
            }

            operator_feedback = "dislike"
            operator_notes = feedback_text

            # ✅ store only when operator submits
            store_embedding(operator_feedback, metadata=feedback_metadata)
            store_embedding(operator_notes, metadata=notes_metadata)

            st.success("Thanks for the feedback! It has been stored successfully.")

            # Reset state
            st.session_state["op_feedback_vote"] = None
            st.session_state["op_feedback_text"] = ""
            st.session_state["operator_feedback"] = ""
            st.session_state["operator_notes"] = ""

# If operator clicked "Useful"
elif st.session_state.get("op_feedback_vote") == "like":

    timestamp = datetime.now(ZoneInfo("Asia/Kolkata"))
    shift = infer_shift(timestamp)
    uid = str(uuid.uuid4())[:8]

    feedback_metadata = {
                "id": f"operator_feedback-{uid}-{timestamp.strftime('%Y-%m-%dT%H-%M-%S')}",
                "namespace": "EBF_operator_feedback",
                "timestamp": timestamp.strftime('%Y-%m-%dT%H-%M-%S'),
                "source": "blast_furnace",
                "shift": shift,
                "date": timestamp.strftime("%Y-%m-%d"),
            }

    notes_metadata = {
                "id": f"operator_notes-{uid}-{timestamp.strftime('%Y-%m-%dT%H-%M-%S')}",
                "namespace": "EBF_operator_notes",
                "timestamp": timestamp.strftime('%Y-%m-%dT%H-%M-%S'),
                "source": "blast_furnace",
                "shift": shift,
                "date": timestamp.strftime("%Y-%m-%d"),
        }

    operator_feedback = "like"
    operator_notes = ""

    # store immediately when operator clicks like
    store_embedding(operator_feedback, metadata=feedback_metadata)
    store_embedding(operator_notes, metadata=notes_metadata)

    st.info("Thanks for confirming this report was useful.")
    st.session_state["op_feedback_vote"] = None

