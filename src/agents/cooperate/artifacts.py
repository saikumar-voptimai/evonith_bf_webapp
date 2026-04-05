"""Artifacts panel — renders the Plot and Data tabs from session_state.

Completely stateless: reads only ``st.session_state``.
Call ``render_artifacts_panel()`` inside any column context.
"""

import streamlit as st


def render_artifacts_panel() -> None:
    """Render the Artifacts panel (📊 Plot / 📋 Data tabs)."""
    st.subheader("Artifacts")
    plot_tab, data_tab = st.tabs(["📊 Plot", "📋 Data"])

    with plot_tab:
        fig = st.session_state.get("copilot_fig")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key="artifact_plot_main")
        else:
            st.info("No plot yet — ask me to visualize something.")

    with data_tab:
        df   = st.session_state.get("copilot_df")
        meta = st.session_state.get("copilot_df_meta", {})
        if df is not None:
            if meta:
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows",    df.shape[0])
                c2.metric("Columns", df.shape[1])
                c3.metric("Dataset", meta.get("dataset_id", "—"))
            st.dataframe(df, use_container_width=True, key="artifact_df_main")
        else:
            st.info("No data yet — ask me to fetch something.")
