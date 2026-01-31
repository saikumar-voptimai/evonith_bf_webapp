# memory/aggregation.py
# Purpose: Automatic aggregation trigger for day / week / bi-week summaries

from datetime import date
from typing import Optional, Dict

from FurnaceMind.memory.schemas import ShiftSummary
from FurnaceMind.memory.structured_store import StructuredStore
from FurnaceMind.core.contextual_analyzer import ContextualAnalyzer
from FurnaceMind.memory.vector_store import QdrantVectorStore
from FurnaceMind.llm.llm_client import OpenAIClient

from FurnaceMind.utils.payload_helpers import (
    build_day_payload,
    build_week_payload,
    build_biweek_payload,
)


def _get_day_id(shift: ShiftSummary) -> str:
    return shift.shift_start.date().isoformat()


def _get_week_id(d: date) -> str:
    iso_year, iso_week, _ = d.isocalendar()
    return f"{iso_year}-W{iso_week:02d}"


def _get_biweek_id(week_id: str) -> str:
    year_str, w_str = week_id.split("-W")
    year = int(year_str)
    week = int(w_str)

    biweek_idx = (week + 1) // 2
    return f"{year}-BW{biweek_idx:02d}"


def run_aggregation_if_ready(
    *,
    new_shift: ShiftSummary,
    store: StructuredStore,
    schemas: Dict[str, dict],
    vector_store: Optional[QdrantVectorStore] = None,
    shifts_per_day: int = 3,
    days_per_week: int = 7,
):
    """
    Automatically trigger daily → weekly → bi-weekly aggregation
    when sufficient data exists.

    Aggregation is stateless and UI-independent.
    """

    # Create analyzer locally (stateless, safe)
    analyzer = ContextualAnalyzer(OpenAIClient())


    # DAILY AGGREGATION (3 shifts → 1 day)
    day_id = _get_day_id(new_shift)
    shifts_today = store.get_shifts_for_day(day_id)

    if len(shifts_today) >= shifts_per_day and not store.daily_exists(day_id):
        shift_payloads = [
            {
                "window_id": s.shift_id,
                "start_time": s.shift_start.isoformat(),
                "end_time": s.shift_end.isoformat(),
                "summary_text": store.get_report(
                    level="shift",
                    window_id=s.shift_id,
                )["structured"]["summary_text"],

                # Furnace Stability Index (safe for legacy shifts)
                "stability_status": (
                    s.stability_status if s.stability_index is not None else "UNKNOWN"
                ),
                "stability_index": s.stability_index,
            }
            for s in shifts_today
        ]

        daily_text, daily_structured = analyzer.build_day_summary(
            day_id=day_id,
            shift_payloads=shift_payloads,
        )

        store.save_daily_summary(
            day_id=day_id,
            summary_text=daily_text,
            structured=daily_structured,
        )

        if vector_store is not None:
            payload = build_day_payload(
                day_id=day_id,
                shift_payloads=shift_payloads,
                structured_summary=daily_structured,
                llm_text=daily_text,
                schema=schemas["day"],
            )

            vector_store.add_window(
                window_id=payload["window_id"],
                embedding_text=payload["summary_text"],
                payload=payload,
            )


    # WEEKLY AGGREGATION (7 days → 1 week)
    week_id = _get_week_id(new_shift.shift_start.date())
    daily_for_week = store.get_daily_for_week(week_id)

    if len(daily_for_week) >= days_per_week and not store.weekly_exists(week_id):
        day_payloads = [
            {
                "window_id": d["window_id"],
                "start_time": d["start_time"],
                "end_time": d["end_time"],
                "summary_text": d["summary_text"],
            }
            for d in daily_for_week
        ]

        weekly_text, weekly_structured = analyzer.build_week_summary(
            week_id=week_id,
            day_payloads=day_payloads,
        )

        store.save_weekly_summary(
            week_id=week_id,
            summary_text=weekly_text,
            structured=weekly_structured,
        )

        if vector_store is not None:
            payload = build_week_payload(
                week_id=week_id,
                day_payloads=day_payloads,
                structured_summary=weekly_structured,
                llm_text=weekly_text,
                schema=schemas["week"],
            )

            vector_store.add_window(
                window_id=payload["window_id"],
                embedding_text=payload["summary_text"],
                payload=payload,
            )


    # BI-WEEKLY AGGREGATION (2 weeks → 1 bi-week)
    biweek_id = _get_biweek_id(week_id)
    weeks_for_biweek = store.get_weeks_for_biweek(biweek_id)

    if len(weeks_for_biweek) >= 2 and not store.biweekly_exists(biweek_id):
        week_payloads = [
            {
                "window_id": w["window_id"],
                "start_time": w["start_time"],
                "end_time": w["end_time"],
                "summary_text": w["summary_text"],
            }
            for w in weeks_for_biweek
        ]

        biweek_text, biweek_structured = analyzer.build_biweek_summary(
            biweek_id=biweek_id,
            week_payloads=week_payloads,
        )

        store.save_biweekly_summary(
            biweek_id=biweek_id,
            summary_text=biweek_text,
            structured=biweek_structured,
        )

        if vector_store is not None:
            payload = build_biweek_payload(
                biweek_id=biweek_id,
                week_payloads=week_payloads,
                structured_summary=biweek_structured,
                llm_text=biweek_text,
                schema=schemas["biweek"],
            )

            vector_store.add_window(
                window_id=payload["window_id"],
                embedding_text=payload["summary_text"],
                payload=payload,
            )
