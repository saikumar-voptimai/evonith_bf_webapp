"""Furnace commentary on a stored optimiser run (SV-3.2, #299): page 11's commentary as a worker job.

The context is page 11's (``utils.burden_engine.commentary.build_context``): the optimiser notes
(``assets/burden_engine/optimiser_notes.md``), the fact register as a table and the run, rebuilt here from the run's
stored case (``optimiser_run.inputs``: settings, limits, calibration, regime, yard) and result. The model check is
page 11's Streamlit composition and is not rebuilt here: the context lists it as not available, as page 11 does when
the Coke Energy pipeline cannot run. The model is called with ``SYSTEM_PROMPT`` through page 7's OpenRouter client,
tried twice (``LLM_TRIES``, ``LLM_RETRY_DELAY_S`` apart); a commentary that fails twice is stored as ``failed`` with
the error of each try.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from services.optimiser_results import flux_ids_of
from utils.burden_engine.commentary import SYSTEM_PROMPT, CommentaryContext, build_context, load_notes
from utils.burden_engine.facts import FactRegister, load_register

log = logging.getLogger(__name__)

#: Time limit of one LLM call, s (the SDK's own retries are off: ``LLM_TRIES`` counts every call).
LLM_TIMEOUT_S = 120.0
#: LLM calls per commentary: the first and one retry.
LLM_TRIES = 2
#: Wait before the retry, s.
LLM_RETRY_DELAY_S = 15.0
#: Page 11's mode labels (``MODES`` of the page).
MODE_LABELS = {
    "cheapest": "Step 1 · Cheapest production level",
    "fixed": "Step 2 · Fixed production",
    "max": "Maximum production",
}
THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

Progress = Callable[[float, str], None]


def _get(row: Any, name: str, default: Any = None) -> Any:
    return row.get(name, default) if isinstance(row, Mapping) else getattr(row, name, default)


def _frame(rows: Any) -> pd.DataFrame | None:
    return pd.DataFrame(list(rows)) if rows else None


def run_context(
    inputs: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    notes: str | None = None,
    reg: FactRegister | None = None,
    model_check: Mapping[str, Any] | None = None,
) -> CommentaryContext:
    """Page 11's commentary context for a stored run: its plan (Step 2 with fuel, Step 1 the cheapest output, else
    the maximum production), today's charged blend and the ore + flux-only answer (Step 2), the value of limits, the
    frontier (Step 1) and the yard, under the notes and the fact register."""
    mode = result.get("mode")
    if mode not in MODE_LABELS:
        raise ValueError(f"not an optimiser result (mode {mode!r})")
    plan, today, ore_only, frontier, label = None, None, None, None, ""
    if mode == "fixed":
        plan, today, ore_only = result.get("with_fuel"), result.get("today"), result.get("ore_only")
        label = f"fixed production {plan['P_tday']:,.0f} t/d" if plan else ""
    elif mode == "cheapest":
        plan = result.get("best")
        frontier = dict(result, frontier=_frame(result.get("frontier")) if result.get("frontier") else pd.DataFrame())
        label = f"cheapest output on the Step 1 frontier, {plan['P_tday']:,.0f} t/d" if plan else ""
    else:
        plan = result.get("result")
        label = f"maximum production {plan['P_tday']:,.0f} t/d" if plan else ""
    cal = dict(inputs.get("calibration") or {})
    yard = dict(inputs.get("yard") or {})
    return build_context(
        notes=load_notes() if notes is None else notes,
        reg=reg or load_register(),
        mode=MODE_LABELS[mode],
        plan=plan,
        plan_label=label,
        names=dict(inputs.get("materials") or {}),
        flux_ids=flux_ids_of(plan) if plan else [],
        settings=(plan or {}).get("settings") or inputs.get("settings") or {},
        limits=(plan or {}).get("limits") or inputs.get("limits") or {},
        yard_source=str(inputs.get("yard_source") or "reference"),
        calibration=dict(
            source=cal.get("source"),
            window=cal.get("window", ""),
            omega=cal.get("omega"),
            Q_LZ_GJh=cal.get("Q_LZ_GJh"),
            Q_UZ_GJh=cal.get("Q_UZ_GJh"),
            message=cal.get("message", "") if cal.get("source") != "live" else "",
        ),
        regime=inputs.get("regime"),
        target_t_day=inputs.get("production_target_t_d") if mode == "fixed" else None,
        model_check=model_check,
        today=today,
        today_window=str(inputs.get("today_window") or "over the last 24 h"),
        ore_only=ore_only,
        value_of_limits=_frame(result.get("value_of_limits")) if mode == "fixed" else None,
        frontier=frontier,
        materials=list(yard.get("materials") or []),
        fluxes=list(yard.get("fluxes") or []),
        fuel=yard.get("fuel") or {},
    )


def commentary_client() -> Any:
    """Page 7's LLM client (``OpenRouterClient``) with the time limit ``LLM_TIMEOUT_S`` and no SDK retries. Raises
    when the client cannot be built (``OPENROUTER_API_KEY`` not set)."""
    from agents.llm.llm_client import OpenRouterClient

    client = OpenRouterClient()
    sdk = getattr(client, "client", None)
    if sdk is not None and hasattr(sdk, "with_options"):
        client.client = sdk.with_options(timeout=LLM_TIMEOUT_S, max_retries=0)
    return client


def _short(exc: BaseException, limit: int = 300) -> str:
    text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    return text if len(text) <= limit else text[: limit - 1] + "…"


def write(
    context: CommentaryContext,
    *,
    client_factory: Callable[[], Any] | None = None,
    tries: int | None = None,
    retry_delay_s: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """The commentary on ``context``: status ok with the text (reasoning blocks removed), the try and the model, or
    failed with the error of each try."""
    tries = LLM_TRIES if tries is None else tries
    retry_delay_s = LLM_RETRY_DELAY_S if retry_delay_s is None else retry_delay_s
    try:
        client = (client_factory or commentary_client)()
    except Exception as exc:
        log.warning("optimiser commentary: no LLM client (%s)", _short(exc))
        return dict(status="failed", text=None, tries=0, error=f"no LLM client: {_short(exc)}", model=None)
    errors: list[str] = []
    for attempt in range(1, tries + 1):
        if progress is not None:
            progress(0.2 + 0.7 * (attempt - 1) / tries, f"Writing the commentary (try {attempt} of {tries})")
        try:
            text = THINK.sub("", str(client.generate(SYSTEM_PROMPT, context.text) or "")).strip()
            if not text:
                raise ValueError("the LLM returned an empty commentary")
            model = getattr(client, "last_actual_model", None) or getattr(client, "model", None)
            return dict(
                status="ok", text=text, tries=attempt, error=None, model=model if isinstance(model, str) else None
            )
        except Exception as exc:
            errors.append(f"try {attempt}: {_short(exc)}")
            log.warning("optimiser commentary, try %d of %d failed: %s", attempt, tries, _short(exc))
            if attempt < tries and retry_delay_s > 0:
                sleep(retry_delay_s)
    return dict(status="failed", text=None, tries=tries, error="; ".join(errors), model=None)


def generate(
    run: Any,
    *,
    client_factory: Callable[[], Any] | None = None,
    retry_delay_s: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    progress: Progress | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The commentary on a stored run (a ``twin.optimiser_run`` row or a mapping of its columns) as the values of a
    ``twin.optimiser_commentary`` row (without ``job_id`` and ``requested_by``). Raises ``ValueError`` for a run
    without a plan."""
    status = _get(run, "status")
    if status != "ok":
        raise ValueError(f"the run has no plan to comment on (status {status})")
    notes, reg = load_notes(), load_register()
    context = run_context(dict(_get(run, "inputs") or {}), dict(_get(run, "result") or {}), notes=notes, reg=reg)
    if not context.is_usable:
        raise ValueError("the run has no plan to comment on")
    note = write(context, client_factory=client_factory, retry_delay_s=retry_delay_s, sleep=sleep, progress=progress)
    note["body"] = note.pop("text")
    return dict(
        run_id=_get(run, "id"),
        generated_at=now or datetime.now(UTC),
        context=context.text,
        context_sha1=hashlib.sha1(context.text.encode("utf-8")).hexdigest(),
        notes_sha1=hashlib.sha1(notes.encode("utf-8")).hexdigest() if notes else None,
        facts_version=str(reg.version),
        missing=list(context.missing),
        **note,
    )


def view(row: Any) -> dict[str, Any]:
    """A stored commentary for the API."""
    return {
        "id": str(_get(row, "id")),
        "run_id": str(_get(row, "run_id")),
        "job_id": _get(row, "job_id"),
        "requested_by": _get(row, "requested_by"),
        "generated_at": _get(row, "generated_at"),
        "status": _get(row, "status"),
        "text": _get(row, "body"),
        "error": _get(row, "error"),
        "tries": _get(row, "tries") or 0,
        "model": _get(row, "model"),
        "missing": list(_get(row, "missing") or []),
        "notes_sha1": _get(row, "notes_sha1"),
        "facts_version": _get(row, "facts_version"),
        "context_sha1": _get(row, "context_sha1"),
        "context": _get(row, "context"),
    }
