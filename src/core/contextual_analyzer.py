from typing import List, Optional
from utils.fm_prompts import PromptTemplates


class ContextualAnalyzer:
    """
    Builds contextual summaries across time horizons
    using stored summaries (memory-based reasoning).
    """

    def __init__(self, llm):
        self.llm = llm


    # Helpers
    @staticmethod
    def _unwrap_payloads(items: list) -> list:
        """
        Normalize retriever results into flat payload dicts.
        """
        normalized = []
        for item in items:
            if isinstance(item, dict) and "payload" in item:
                normalized.append(item["payload"])
            else:
                normalized.append(item)
        return normalized

    @staticmethod
    def _build_history_block(
        previous_shift: Optional[str],
        historical_similar: Optional[List[str]],
    ) -> str:
        """
        Build readable history context block for LLM.
        """
        sections = []

        if previous_shift:
            sections.append(
                "PREVIOUS SHIFT CONTEXT:\n"
                + previous_shift
            )

        if historical_similar:
            sections.append(
                "SIMILAR HISTORICAL SHIFTS:\n"
                + "\n\n".join(historical_similar)
            )

        return "\n\n".join(sections)


    # Day summary (UPDATED)

    def build_day_summary(
    self,
    day_id: str,
    shift_payloads: list,
    *,
    previous_shift: Optional[str] = None,
    historical_similar: Optional[List[str]] = None,
    operator_notes: Optional[List[str]] = None, 
):
        """
        Build a context-aware daily summary using:
        - current shift(s)
        - previous shift (optional)
        - similar historical shifts (optional)
        - operator-reported observations (optional)
        """

        shift_payloads = self._unwrap_payloads(shift_payloads)

        # Current shift block
        current_block = "\n\n".join(
            f"{p.get('shift_name', 'Unknown Shift')} "
            f"({p.get('start_time', '?')}–{p.get('end_time', '?')}):\n"
            f"{p.get('summary_text', '')}"
            for p in shift_payloads
        )

        # Historical context block
        history_block = self._build_history_block(
            previous_shift=previous_shift,
            historical_similar=historical_similar,
        )

        # Operator observations block (NEW)
        operator_block = None
        if operator_notes:
            operator_block = (
                "Operator-Reported Observations:\n"
                + "\n".join(f"- {note}" for note in operator_notes)
            )

        # Final context
        full_context = "\n\n".join(
            block
            for block in [history_block, operator_block, current_block]
            if block
        )

        user_prompt = PromptTemplates.CONTEXTUAL_ANALYSIS_TASK.format(
            current_shift_summary=current_block,
            previous_shift_summary=previous_shift or "No previous shift data available.",
            historical_summaries=(
                "\n\n".join(historical_similar)
                if historical_similar
                else "No similar historical cases found."
            ),
            operator_observations=(
                "\n".join(f"- {note}" for note in operator_notes)
                if operator_notes
                else "No operator observations reported."
            ),
        )

        llm_summary = self.llm.generate(
            system_prompt=PromptTemplates.CONTEXTUAL_ANALYZER_SYSTEM,
            user_prompt=user_prompt,
        )

        unstable = [
            p for p in shift_payloads
            if p.get("overall_stability") != "stable"
        ]

        structured = {
            "window_id": day_id,
            "overall_stability": "unstable" if unstable else "stable",
            "num_shifts": len(shift_payloads),
            "num_unstable_shifts": len(unstable),
        }

        return llm_summary, structured



    # Week summary (UNCHANGED)
    def build_week_summary(self, week_id: str, day_payloads: list):
        day_payloads = self._unwrap_payloads(day_payloads)

        context_block = "\n\n".join(
            f"Day {p.get('window_id', '?')}:\n{p.get('summary_text', '')}"
            for p in day_payloads
        )

        user_prompt = PromptTemplates.WEEKLY_REPORT_TASK.format(
            weekly_shift_summaries=context_block,
        )

        llm_summary = self.llm.generate(
            system_prompt=PromptTemplates.CONTEXTUAL_ANALYZER_SYSTEM,
            user_prompt=user_prompt,
        )

        unstable_days = [
            p for p in day_payloads
            if p.get("overall_stability") != "stable"
        ]

        structured = {
            "window_id": week_id,
            "overall_stability": "unstable" if unstable_days else "stable",
            "num_days": len(day_payloads),
            "num_unstable_days": len(unstable_days),
        }

        return llm_summary, structured


    # Bi-week summary (UNCHANGED)
    def build_biweek_summary(self, biweek_id: str, week_payloads: list):
        week_payloads = self._unwrap_payloads(week_payloads)

        context_block = "\n\n".join(
            f"Week {p.get('window_id', '?')}:\n{p.get('summary_text', '')}"
            for p in week_payloads
        )

        user_prompt = PromptTemplates.BIWEEKLY_REPORT_TASK.format(
            biweekly_shift_summaries=context_block,
        )

        llm_summary = self.llm.generate(
            system_prompt=PromptTemplates.CONTEXTUAL_ANALYZER_SYSTEM,
            user_prompt=user_prompt,
        )

        unstable_weeks = [
            p for p in week_payloads
            if p.get("overall_stability") != "stable"
        ]

        structured = {
            "window_id": biweek_id,
            "overall_stability": "unstable" if unstable_weeks else "stable",
            "num_weeks": len(week_payloads),
            "num_unstable_weeks": len(unstable_weeks),
        }

        return llm_summary, structured