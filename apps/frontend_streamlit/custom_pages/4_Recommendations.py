from pathlib import Path

import streamlit as st

from config.frontend_settings import is_backend_api_enabled
from services.api_errors import FrontendApiError
from services.recommendations_api import run_recommendations


def _api_token() -> str | None:
    token = str(st.session_state.get("auth_access_token") or "").strip()
    return token or None


def _render_api_mode() -> None:
    st.markdown(
        """
        <h1 style="text-align: center; font-family: 'Times New Roman', Times, serif; ">
            V-Sense - Blast Parameter Optimisation
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Recommendations mode: Backend API")
    signal_text = st.text_area(
        "Signals",
        placeholder="Example:\nPCI_KG/THM=8\nCOKE RATE KG/THM=-5",
        height=140,
    )
    max_items = st.number_input("Max recommendations", min_value=1, max_value=50, value=10)
    if st.button("Run Recommendations", type="primary"):
        signals: dict[str, float] = {}
        for line in signal_text.splitlines():
            if "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            try:
                signals[key.strip()] = float(raw_value.strip())
            except ValueError:
                continue
        try:
            result = run_recommendations(
                {
                    "source": "input_data",
                    "input_data": {"signals": signals},
                    "max_items": int(max_items),
                },
                token=_api_token(),
            )
        except FrontendApiError as exc:
            request_id = f" Request ID: {exc.request_id}." if exc.request_id else ""
            st.error(f"Backend recommendations failed: {exc.message}.{request_id}")
            return
        for item in result.get("items") or []:
            st.subheader(item.get("title") or "Recommendation")
            st.write(item.get("description") or "")
            if item.get("actions"):
                st.write(item["actions"])


if is_backend_api_enabled("recommendations"):
    _render_api_mode()
    st.stop()

import joblib
import numpy as np
import pandas as pd
import yaml

from config.config_loader import load_config
from utils.recommendations.data import DataframesProcessor
from utils.recommendations.llm import call_llm
from utils.recommendations.optimiser import run_optimiser
from utils.recommendations.bounds import load_control_bounds, save_control_bounds
from utils.recommendations.prompts import prompt_recommendation_system

config = load_config()
config_vsense = load_config("setting_vsense.yml")
CONFIG_PATH = Path("src/config/setting_vsense.yml")

# Load external CSS file
css_path = (
    Path(__file__).resolve().parents[1] / "assets" / "css" / "recommendation_style.css"
)

# ── dataset auto-refresh ───────────────────────────────────────────────────
from utils.dataset_refresher import get_version as _ds_get_version, maybe_refresh as _ds_maybe_refresh  # noqa: E402

if _ds_maybe_refresh(config):
    st.sidebar.caption("⏳ Refreshing dataset in background…")

# If a new dataset version landed since this session was started, force
# DataframesProcessor to re-initialise with the fresh static dataset on next run.
_current_ds_version = _ds_get_version()
if st.session_state.get("_ds_version") != _current_ds_version:
    st.session_state.pop("dfprocessor", None)
    st.session_state.pop("df_hist", None)
    st.session_state.pop("df_full", None)
    st.session_state["_ds_version"] = _current_ds_version

if "df_hist" not in st.session_state:
    st.session_state["df_hist"] = None
if "df_full" not in st.session_state:
    st.session_state["df_full"] = None
if "dfprocessor" not in st.session_state:
    st.session_state["dfprocessor"] = None

# Section 0: Set the title page configuration
st.markdown(
    """
    <h1 style="text-align: center; font-family: 'Times New Roman', Times, serif; ">
        V-Sense — Blast Parameter Optimisation
    </h1>
    """,
    unsafe_allow_html=True,
)

st.divider()

debug_on = st.sidebar.toggle("Debug", value=False)
run_llm = st.sidebar.toggle("Run LLM Analysis", key="run_llm_button")
new_steps = 30 if debug_on else 30

if config_vsense.get("OPTIM_STEPS") != new_steps:
    config_vsense["OPTIM_STEPS"] = new_steps
    yaml.safe_dump(config_vsense, open(CONFIG_PATH, "w"))

# Section 1: Select the optimisation type
optimisation_type = st.selectbox(
    "Select Optimisation Type", list(config_vsense["Optimisation"].keys())
)
# Extract input parameter groups for the selected optimisation type
input_params = config_vsense["Optimisation"][optimisation_type]["input_params"]

outputs = [
    config_vsense["Optimisation"][model]["output_param"]
    for model in list(config_vsense["Optimisation"].keys())
]
models_dict = config_vsense["Optimisation"]
# Load the configuration and model
for model in models_dict.keys():
    ip_flat = [
        val for group in models_dict[model]["input_params"].values() for val in group
    ]
    models_dict[model]["input_params_flat"] = ip_flat
    models_dict[model]["Optimised"] = False
    relpath = models_dict[model]["model"]
    model_path = Path(__file__).resolve().parents[2] / relpath
    models_dict[model]["LoadedMLModel"] = joblib.load(model_path)
    if model == optimisation_type:
        models_dict[model]["Optimised"] = True

if st.session_state["df_full"] is None:
    dfprocessor = DataframesProcessor(
        config=config, config_vsense=config_vsense, debug_on=debug_on
    )
    df_full = dfprocessor.df_full
    df_hist = dfprocessor.df_hist
    st.session_state["df_full"] = df_full
    st.session_state["df_hist"] = df_hist
    st.session_state["dfprocessor"] = dfprocessor

# Set the target output based on the optimisation type
target_output = config_vsense["Optimisation"][optimisation_type]["output_param"]

st.subheader("Control Parameters")
persisted_bounds = load_control_bounds()

if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

ip_flat_list = models_dict[optimisation_type]["input_params_flat"]
cp_list = models_dict[optimisation_type]["control_params"]
op_list = [
    config_vsense["Optimisation"][model]["output_param"]
    for model in list(config_vsense["Optimisation"].keys())
]

include_control = {}
if "control_params" not in st.session_state:
    st.session_state["control_params"] = include_control


last_time = st.session_state.df_full.index[-1].tz_convert("Asia/Kolkata")
st.write(f"##### From window ending: {last_time.strftime('%Y-%m-%d %H:%M:%S')}")
with st.form(f"Control Params Form"):
    cols = st.columns(3)
    i = 0
    for cp in cp_list:
        with cols[i % 3]:
            # Get machine limits
            col_min = float(st.session_state.df_full[cp].min())
            col_max = float(st.session_state.df_full[cp].max())
            val = float(st.session_state.df_full[cp].iloc[-1])

            # Use persisted values if any
            cp_min = persisted_bounds.get(cp, {}).get("min", col_min)
            cp_max = persisted_bounds.get(cp, {}).get("max", col_max)
            override_prev = persisted_bounds.get(cp, {}).get("override", False)

            col1, col2 = st.columns([0.05, 0.95])

            # Checkbox in the first column
            with col1:
                override = st.checkbox(" ", value=override_prev, key=f"override_{cp}")

            # Heading in the second column, styled
            col2.markdown(
                f"<div class='param-title'>{cp}</div>", unsafe_allow_html=True
            )

            # Layout for inputs
            # -------- Row 1: Value --------
            val = st.number_input(
                "Value",
                min_value=cp_min,
                max_value=cp_max,
                value=np.clip(val, cp_min, cp_max),
                key=f"val_{cp}",
            )

            # -------- Row 2: Min & Max --------
            min_col, max_col = st.columns(2)

            disabled_chg = True if override else False

            with min_col:
                min_val = st.number_input(
                    "Min",
                    value=cp_min,
                    key=f"min_{cp}",
                    disabled=disabled_chg,
                )

            with max_col:
                max_val = st.number_input(
                    "Max",
                    value=cp_max,
                    key=f"max_{cp}",
                    disabled=disabled_chg,
                )

            # Ensure current value is within limits
            val = min(max(val, min_val), max_val)
            include_control[cp] = {
                "min": min_val,
                "max": max_val,
                "value": val,
                "override": override,
            }

        i += 1
    st.session_state["control_params"] = include_control
    submit_cp = st.form_submit_button("Submit CP & Save bounds")
    if submit_cp:
        st.session_state['control_params'] = include_control
        merged_bounds = persisted_bounds.copy()
        merged_bounds.update(include_control)
        save_control_bounds(merged_bounds)
        st.success("✅ Bounds saved successfully!")

# User-specified input variables:
latest_row = st.session_state.dfprocessor.fetch_live_rm_data()
latest_row.name = latest_row.name.tz_localize("Asia/Kolkata")
with st.expander(
    f"Input Params at {latest_row.name.strftime('%Y-%m-%d %H:%M:%S')}  - Click to expand and override"
):

    cols = st.columns(3)
    raw_mtrl_input = {}
    with st.form(key="Raw Material Input Form"):
        for i, (group_name, params) in enumerate(input_params.items()):
            with cols[i % 3]:
                st.write(f"### {group_name}")
                for param in params:
                    if group_name == "Burden" and param in st.session_state.df_hist:
                        latest_row[param] = st.session_state.df_hist.iloc[-1][param]
                    if param in latest_row and (
                        "CHARGES/HRS." in param or "GEOMIN TYPE" in param
                    ):
                        latest_row[param] = np.round(latest_row[param])
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

        submit_ip = st.form_submit_button("Submit Input Params")
        if submit_ip:
            st.success("✅ Input parameters recorded.")
with st.form("Optimiser Form"):
    cols = st.columns(2)
    with cols[0]:
        lambda_reg = st.slider(
            "Regularisation Parameter (Lambda)",
            min_value=0.0,
            max_value=0.5,
            value=config_vsense["LAMBDA_REG"],
            step=0.005,
            help="Regularisation parameter for the optimisation algorithm.",
        )
    submit_optim_params = st.form_submit_button("Run Optimiser")
    if submit_optim_params:
        st.success("✅ Optimiser run requested.")
        fixed_cp = {
            cp: cp_props["value"]
            for cp, cp_props in include_control.items()
            if not np.isnan(cp_props["value"]) and cp_props["override"]
        }
        user_input = {
            param: raw_mtrl_input.get(param, np.nan) for param in ip_flat_list
        }
        feat_vec_target = st.session_state.dfprocessor.process_dataframe(
            scaler_path=models_dict[optimisation_type].get("scaling", None)
        )
        payload = {
            "feat_vec_target": feat_vec_target.to_dict(orient="list"),
            "models_dict": {
                m: {k: v for k, v in model_cfg.items() if k != "LoadedMLModel"}
                for m, model_cfg in models_dict.items()
            },
            "user_input": user_input,
            "fixed_cp": fixed_cp,
            "lambda_reg": lambda_reg,
        }

        optimal_solution = run_optimiser(
            feat_vec_target,
            models_dict,
            user_input,
            fixed_cp,
            st.session_state.dfprocessor,
            lambda_reg=lambda_reg,
            impute_lags=False,
        )

        st.subheader("Optimisation Results")
        # Show metrics for each control parameter and the target output
        cols = st.columns(4)
        j = 0
        for i, (key, new_val) in enumerate(optimal_solution.items()):
            if (
                key in st.session_state.df_full.columns
                and key != target_output
                and key not in outputs
            ):
                with cols[j % 4]:
                    old_val = st.session_state.df_full[key].iloc[-1]
                    delta = new_val - old_val
                    if abs(delta) / abs(old_val) >= 0.01:
                        st.metric(
                            label=key, value=f"{new_val:.2f}", delta=f"{delta:+.2f}"
                        )
                        j += 1
                    if (
                        abs(delta) / abs(old_val) < 0.01
                    ):  # Only show significant changes
                        st.metric(label=key, value=f"{old_val:.2f}", delta=f"{0:+.2f}")
                        j += 1

        st.metric(
            label=target_output,
            value=f"{optimal_solution[target_output + '_current']:.2f}",
            delta=f"{optimal_solution[target_output + '_current'] - optimal_solution[target_output + '_previous']:+.2f}",
        )

        cols = st.columns(3)
        j = 0
        for i, (key, new_val) in enumerate(optimal_solution.items()):
            key_feat = key.replace("_current", "").replace("_previous", "")
            if key_feat in outputs and key_feat != target_output and "_previous" in key:
                with cols[j % 3]:
                    old_val = optimal_solution[key_feat + "_previous"]
                    new_val = optimal_solution[key_feat + "_current"]
                    delta = new_val - old_val
                    st.metric(
                        label=key_feat, value=f"{new_val:.2f}", delta=f"{delta:+.2f}"
                    )
                    j += 1

        # Optional: show dependent variables computed from optimal knobs
        dep_prev_suffix = "_dep_previous"
        dep_curr_suffix = "_dep_current"
        dep_bases = sorted(
            {
                k[: -len(dep_prev_suffix)]
                for k in optimal_solution.keys()
                if k.endswith(dep_prev_suffix)
            }
        )

        if dep_bases:
            with st.expander(
                "Dependent Parameters (Auto-calculated) — optional", expanded=False
            ):
                show_dep = st.checkbox(
                    "Show dependent parameter impact",
                    value=False,
                    key="show_dep_params",
                )
                if show_dep:
                    cols = st.columns(4)
                    j = 0
                    for base in dep_bases:
                        prev_key = base + dep_prev_suffix
                        curr_key = base + dep_curr_suffix
                        if (
                            prev_key not in optimal_solution
                            or curr_key not in optimal_solution
                        ):
                            continue

                        prev_val = float(optimal_solution[prev_key])
                        curr_val = float(optimal_solution[curr_key])
                        delta = curr_val - prev_val
                        with cols[j % 4]:
                            st.metric(
                                label=base,
                                value=f"{curr_val:.3g}",
                                delta=f"{delta:+.3g}",
                            )
                        j += 1

        # Section 4: Generate recommendations using LLM
        st.subheader("Recommendations")
        system = "You are a precise, senior blast furnace advisor. Be concise, numeric, and actionable."
        df_hist = st.session_state.df_hist.drop(columns=target_output)
        prompt = prompt_recommendation_system(
            df_hist=df_hist,
            optimal_solution=optimal_solution,
            target_output=target_output,
        )
        with st.spinner("Generating review…"):
            if run_llm:
                out = call_llm(system, prompt)
                st.markdown(out)
