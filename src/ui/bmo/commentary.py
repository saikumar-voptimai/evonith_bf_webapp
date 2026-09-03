"""The furnace commentary panel: model call, caching, and rendering.

Context assembly lives in ``utils/bmo/commentary.py`` and is pure. This module
does the three things that cannot be unit-tested cheaply - fetch the recent
history, call the model, and put the result on screen.

IT IS ON A BUTTON, NOT AUTOMATIC. A model call costs money and several seconds,
and the operator reruns this page constantly while editing ore rows. Generating
commentary on every rerun would burn tokens on blends nobody is considering.

THE GENERATED TEXT IS LABELLED AS GENERATED. It sits below the numbers it
describes, never above them, and the panel says plainly that it is a model's
reading rather than a calculation. An operator must never be unable to tell
which is which.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import streamlit as st

from utils.bmo.commentary import SYSTEM_PROMPT, build_commentary_context

log = logging.getLogger(__name__)

RECENT_DAYS = 3
_MEASUREMENTS = ["process_params"]


@st.cache_data(ttl=900, show_spinner=False)
def _recent_history(days: int):
    """Hourly online data for the trend window.

    Cached for fifteen minutes: it is the same query for every blend the
    operator tries, and the furnace does not move meaningfully inside that.
    """

    from furnace_data.influx.online import fetch_online_df

    return fetch_online_df(
        selected_measurements=_MEASUREMENTS,
        time_range=f"last {days} days",
        window_by="1 hour",
        column_naming="field",
    )


def _generate(system_prompt: str, user_prompt: str) -> str:
    """One model call. Raises on failure so the caller can report it."""

    from agents.llm.llm_client import get_llm_client

    client = get_llm_client()
    return str(client.generate(system_prompt, user_prompt) or "").strip()


def render_furnace_commentary(
    *,
    live_snapshot: dict[str, Any] | None,
    ores: Sequence[Any] | None,
    lp_blend: Any,
    de_blend: Any,
    manual_blend: Any,
    calibration: Any = None,
    energy_anchor: Any = None,
    production_target_mt: float | None = None,
) -> None:
    """Render the commentary panel for the blends currently on screen."""

    st.markdown("### 🧠 Furnace commentary")
    st.caption(
        "A reading of the numbers above by the FurnaceMind model — the live "
        "furnace state, the last three days, what is in stock, and the blend "
        "being recommended, together with the known defects in each of those "
        "figures. **It is generated text, not a calculation.** Every number it "
        "quotes comes from this page; check anything you intend to act on."
    )

    if lp_blend is None and de_blend is None:
        st.info("Run the optimizer first — there is no blend to comment on yet.")
        return

    with st.spinner("Reading the last three days…"):
        try:
            recent = _recent_history(RECENT_DAYS)
        except Exception as exc:  # noqa: BLE001 - history is optional context
            log.warning("Commentary history unavailable: %s", exc)
            recent = None

    # DE is the page's recommendation when it beat the LP; the guardrail upstream
    # replaces a worse DE result with the LP blend, so a DE result that is not a
    # fallback is genuinely the better one.
    de_is_fallback = bool(
        (getattr(de_blend, "diagnostics", {}) or {}).get("de_fell_back_to_lp")
    )
    recommended_label = (
        "DE total cost" if de_blend is not None and not de_is_fallback
        else "LP baseline"
    )

    context = build_commentary_context(
        live_snapshot=live_snapshot,
        recent_frame=recent,
        ores=ores,
        lp_blend=lp_blend,
        de_blend=de_blend,
        manual_blend=manual_blend,
        recommended_label=recommended_label,
        calibration=calibration,
        energy_anchor=energy_anchor,
        production_target_mt=production_target_mt,
        recent_days=RECENT_DAYS,
    )

    left, right = st.columns([3, 1], vertical_alignment="center")
    with left:
        if context.missing:
            st.caption(
                "⚠️ Not available for this run: " + ", ".join(context.missing)
                + ". The commentary will say so rather than guess."
            )
    with right:
        asked = st.button(
            "Generate commentary",
            width="stretch",
            type="primary",
            key="bmo_commentary_go",
            help="Sends the context below to the model. Takes a few seconds.",
        )

    if asked:
        with st.spinner("Thinking about this blend…"):
            try:
                st.session_state["bmo_commentary"] = _generate(
                    SYSTEM_PROMPT, context.text
                )
                st.session_state["bmo_commentary_context"] = context.text
            except Exception as exc:  # noqa: BLE001 - never break the results page
                log.exception("Furnace commentary failed")
                st.session_state["bmo_commentary"] = ""
                st.error(
                    "Could not generate commentary: "
                    + str(exc).splitlines()[0][:200]
                    + "\n\nThe numbers above are unaffected."
                )

    commentary = st.session_state.get("bmo_commentary")
    if commentary:
        with st.container(border=True):
            st.markdown(commentary)
        st.caption(
            "Generated by a language model from the context below. It can be "
            "wrong, and it cannot see anything that is not in that context — "
            "no burden distribution changes, no maintenance, no what the "
            "previous shift saw."
        )

    with st.expander("What the model was given", expanded=False):
        # Shown in full, deliberately. A commentary an operator cannot audit is
        # a commentary they should not act on, and the most common failure of
        # these panels is a confident sentence built on a figure that was
        # missing.
        st.code(
            st.session_state.get("bmo_commentary_context") or context.text,
            language="text",
        )
