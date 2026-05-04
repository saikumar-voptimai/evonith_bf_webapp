"""Latest-artifact panel — pinned view of the most recent plot and/or dataframe.

Stateless: reads only ``st.session_state``.  Shows both ``fm_fig`` and
``fm_df`` when both are present so the user can inspect the data that
underlies the chart without scrolling back through the conversation.

Call ``render_artifacts_panel()`` inside the right-column context.
"""

import streamlit as st


def render_artifacts_panel() -> None:
    """Render the pinned latest-artifact panel.

    Shows ``fm_fig`` (if set) followed by ``fm_df`` (if set).
    Falls back to a placeholder when neither is available.
    """
    fig = st.session_state.get("fm_fig")
    df = st.session_state.get("fm_df")
    meta = st.session_state.get("fm_df_meta", {})

    if fig is None and (df is None or df.empty):
        st.markdown(
            "<div style='text-align:center;color:#94a3b8;padding:3rem 1rem;"
            "font-size:0.88rem;'>Chart or data will appear here<br>"
            "after your first request.</div>",
            unsafe_allow_html=True,
        )
        return

    if fig is not None:
        st.caption("Latest plot")
        st.plotly_chart(fig, use_container_width=True, key="artifact_plot_pinned")

    if df is not None and not df.empty:
        row_count = df.shape[0]
        col_count = df.shape[1]
        label = meta.get("dataset_id", "")
        st.caption(
            f"Latest data · {row_count:,} rows × {col_count} cols"
            + (f" · {label}" if label else "")
        )
        st.dataframe(
            df.head(100), use_container_width=True,
            hide_index=True, key="artifact_df_pinned",
        )
        st.download_button(
            "⬇ Download CSV",
            df.to_csv(index=True).encode(),
            file_name=f"{label or 'data'}.csv",
            mime="text/csv",
            key="artifact_df_download",
            use_container_width=True,
        )
