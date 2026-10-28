"""Doubles for the FurnaceMind agent tests (SV-5.1, #312): a scripted chat model, synthetic plant data behind the
real tools, vector stores, and the 20 reference questions.

``ScriptedLLM`` answers each reference question with a fixed sequence of tool-call rounds; arguments may depend on
what the model has seen (dataset ids from earlier tool results, the size of the replayed history), so two paths that
replay different histories or produce different tool results also produce different tool calls. It serves the plain
call (page 7) and the streamed call (agent service) from the same script; the streamed call goes through the real
``assemble_stream`` with chunks split inside the ``<think>`` tags.

``install_plant_data`` puts synthetic data behind the real tools (``fetch_ml_data``, ``fetch_online_data``,
``fetch_offline_data``, ``concat_datasets``, ``merge_furnace_data``, ``execute_python_plot``, ...): only the data
sources are replaced, never the tools.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from agents.llm.llm_client import assemble_stream

DATASET_ID = re.compile(r"dataset_id=(\S+)")
TIMESTAMPED_ID = re.compile(r"\b([a-z_]+)_\d{8}T\d{6}Z_(\d+)\b")
NOW_IST = pd.Timestamp("2026-10-21 10:00")  # frozen "now" of the ML-dataset tools (IST, naive)


def normalise(value: Any) -> Any:
    """Dataset ids carry the UTC second of their creation: replace it so two runs compare equal."""
    text = json.dumps(value, sort_keys=True, default=str)
    return json.loads(TIMESTAMPED_ID.sub(r"\1_<ts>_\2", text))


# ------------------------------------------------------------------------------------------------- plant data


def static_dataset() -> pd.DataFrame:
    """Hourly static ML dataset, 1 Sep - 20 Oct 2026 (IST, naive index)."""
    index = pd.date_range("2026-09-01 00:00", "2026-10-20 23:00", freq="1h")
    t = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "ACT. FUEL RATEKG/THM.": 545.0 + 6.0 * np.sin(t / 24.0),
            "COKE RATE KG/THM": 365.0 + 4.0 * np.cos(t / 36.0),
            "FURNACETOPGASANALYSISCO2ETACO": 48.0 + 0.5 * np.sin(t / 12.0),
            "PRODUCTIONTONNESPERHR": 255.0 + 3.0 * np.cos(t / 48.0),
            "HOT METAL SI%": 0.45 + 0.05 * np.sin(t / 8.0),
        },
        index=index,
    )


def online_frame(**_: Any) -> pd.DataFrame:
    """15-min telemetry ending 20 Oct 2026 12:00 UTC: heat loads, the 18660 mm temperatures and the fuel rate."""
    index = pd.date_range("2026-10-20 04:00", "2026-10-20 12:00", freq="15min", tz="UTC")
    t = np.arange(len(index), dtype=float)
    columns: dict[str, Any] = {f"heat_load_row_{r}": 900.0 + 10 * r + np.sin(t + r) for r in range(6, 11)}
    columns.update({f"heat_load_row6_10_q{q}": 220.0 + q + np.cos(t / 2 + q) for q in range(1, 5)})
    columns.update({f"temp_18660_{c}": 140.0 + i * 3 + np.sin(t / 3 + i) for i, c in enumerate("abcd")})
    columns["fuel_rate"] = 548.0 + np.sin(t / 4)
    return pd.DataFrame(columns, index=index)


def offline_frame(**_: Any) -> pd.DataFrame:
    """Hourly hot-metal analysis ending 20 Oct 2026 12:00 UTC."""
    index = pd.date_range("2026-10-19 12:00", "2026-10-20 12:00", freq="1h", tz="UTC")
    t = np.arange(len(index), dtype=float)
    return pd.DataFrame({"chem_pct_si": 0.42 + 0.03 * np.sin(t / 5), "hmt_c": 1480 + 5 * np.cos(t / 6)}, index=index)


class ShiftStore:
    """Vector store double of the shift summaries; a query with ``explode`` raises (tool failure path)."""

    def search_similar_windows(self, *, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if "explode" in query_text:
            raise RuntimeError("qdrant timeout")
        return [
            {"payload": {"window_id": f"2026-10-1{i}_Shift_A", "summary_text": f"Stable shift {i} for: {query_text}"}}
            for i in range(min(top_k, 2))
        ]


class KnowledgeStore:
    """Knowledge store double: two text chunks, no visual evidence."""

    collection_name = "test_knowledge"

    def search(self, query: str, **_: Any) -> list[dict[str, Any]]:
        return [
            {"score": 0.82, "payload": {"content": f"Tuyere SOP section on {query}", "source": "sop.pdf", "page": 3}},
            {"score": 0.61, "payload": {"content": "Water leak isolation steps", "source": "sop.pdf", "page": 4}},
        ]


def resources() -> dict[str, Any]:
    """What page 7 puts into the session (and the service seeds into each conversation's tool state)."""
    return {"shift_store": ShiftStore(), "knowledge_store": KnowledgeStore()}


def install_plant_data(monkeypatch: Any, tmp_path: Any) -> None:
    """Synthetic sources behind the real tools; tool errors go to ``tmp_path`` (not the tracked tool_errors.md)."""
    import agents.furnace_tools as ft

    monkeypatch.setattr(ft, "load_static_dataset", static_dataset)
    monkeypatch.setattr(ft, "fetch_online_df", online_frame)
    monkeypatch.setattr(ft, "_fetch_offline_report_df", offline_frame)
    monkeypatch.setattr(ft, "_now_ist_naive", lambda: NOW_IST)
    monkeypatch.setattr(ft, "_TOOL_ERRORS_PATH", tmp_path / "tool_errors.md")
    ft._static_dataset_outside_streamlit.clear()


def record_dispatch(monkeypatch: Any) -> list[tuple[str, dict[str, Any]]]:
    """Every tool call the agent dispatches, in order (the real dispatcher still runs)."""
    import agents.furnace_tools as ft

    calls: list[tuple[str, dict[str, Any]]] = []
    real = ft.execute_openai_tool_call

    def recording(*, name: str, arguments: dict[str, Any]) -> str:
        calls.append((name, dict(arguments)))
        return real(name=name, arguments=arguments)

    monkeypatch.setattr(ft, "execute_openai_tool_call", recording)
    return calls


# --------------------------------------------------------------------------------------------- the scripted model

Args = dict[str, Any] | Callable[[list[dict[str, Any]]], dict[str, Any]]
Round = list[tuple[str, Args]]

PLOT_FUEL = "fig = px.line(df.reset_index(), x=df.index.name or 'index', y=df.columns[0], title='Fuel rate')"
PLOT_BAR = "fig = px.bar(df.reset_index(), x=df.index.name or 'index', y=df.columns[0], title='Fuel rate (bar)')"
PLOT_LAST = (
    "fig = go.Figure()\nfor c in df.columns[:3]:\n    fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c))\n"
    "fig.update_layout(title='First three columns')"
)


def turn_ids(messages: list[dict[str, Any]]) -> list[str]:
    """Dataset ids returned by the tools since the last user question, in order."""
    last_user = max(i for i, m in enumerate(messages) if m.get("role") == "user" and isinstance(m.get("content"), str))
    ids: list[str] = []
    for m in messages[last_user:]:
        if m.get("role") == "tool":
            ids.extend(DATASET_ID.findall(str(m.get("content"))))
    return ids


def context_size(messages: list[dict[str, Any]]) -> int:
    """Replayed history: user and assistant texts before the last question (system prompt excluded)."""
    last_user = max(i for i, m in enumerate(messages) if m.get("role") == "user" and isinstance(m.get("content"), str))
    return sum(1 for m in messages[:last_user] if m.get("role") in {"user", "assistant"} and not m.get("tool_calls"))


#: The 20 reference questions: question -> tool-call rounds (``FAIL`` = the model is unavailable for that question).
FAIL = "fail"
REFERENCE: list[tuple[str, list[Round] | str]] = [
    (
        "Plot the fuel rate for the first week of October.",
        [
            [("fetch_ml_data", {"start_time": "2026-10-01", "end_time": "2026-10-07", "columns": ["fuel rate"]})],
            [("execute_python_plot", {"code": PLOT_FUEL})],
        ],
    ),
    (
        "Show the heat loads of the last 8 hours.",
        [
            [
                (
                    "fetch_online_data",
                    {"lookback": "8h", "measurement_groups": ["heatload_delta_t", "temperature_profile"]},
                )
            ],
            [("render_heatload_plot", {})],
        ],
    ),
    ("Load shift A of 10 October.", [[("load_static_shift_data", {"shift_date": "2026-10-10", "shift_label": "A"})]]),
    (
        "Stitch the static data with the live data since 18 October.",
        [
            [("fetch_ml_data", {"start_time": "2026-10-18", "columns": ["fuel"]})],
            [("fetch_online_data", {"lookback": "1d", "measurement_groups": ["process_params"]})],
            [("concat_datasets", lambda m: {"dataset_ids": turn_ids(m)[:2]})],
        ],
    ),
    (
        "Which earlier shifts were unstable?",
        [[("search_shift_history", lambda m: {"query": f"unstable shifts ctx={context_size(m)}"})]],
    ),
    ("What does the SOP say about tuyere leaks?", [[("search_knowledge_docs", {"query": "tuyere leak"})]]),
    (
        "Merge the live data with the hot metal analysis.",
        [
            [
                ("fetch_online_data", {"lookback": "8h", "measurement_groups": ["process_params"]}),
                ("fetch_offline_data", {"report_type": "HM_SLAG", "lookback_days": 2}),
            ],
            [
                (
                    "merge_furnace_data",
                    lambda m: {"online_dataset_id": turn_ids(m)[0], "offline_dataset_ids": [turn_ids(m)[1]]},
                )
            ],
        ],
    ),
    ("Explain RAFT in one paragraph.", []),
    ("Plot with broken arguments.", [[("execute_python_plot", "{not json")]]),
    ("Use the unknown tool.", [[("fetch_everything", {"all": True})]]),
    (
        "Fetch September and October separately and join them.",
        [
            [
                ("fetch_ml_data", {"start_time": "2026-09-25", "end_time": "2026-09-30", "resample": "8h"}),
                ("fetch_ml_data", {"start_time": "2026-10-01", "end_time": "2026-10-05", "resample": "8h"}),
            ],
            [("concat_datasets", lambda m: {"dataset_ids": turn_ids(m)})],
        ],
    ),
    (
        "Which columns does the dataset have? Then plot them.",
        [
            [("execute_python_plot", {"code": "print(sorted(df.columns.tolist()))"})],
            [("execute_python_plot", {"code": PLOT_LAST})],
        ],
    ),
    ("Run this code: import os", [[("execute_python_plot", {"code": "import os\nfig = None"})]]),
    ("This question finds the model busy.", FAIL),
    ("Search the shifts until the store explodes.", [[("search_shift_history", {"query": "explode now"})]]),
    ("Plot that again as a bar chart.", [[("execute_python_plot", {"code": PLOT_BAR})]]),
    (
        "Daily coke rate of September, then a look at its columns, then the plot.",
        [
            [("fetch_ml_data", {"start_time": "2026-09-01", "end_time": "2026-09-30", "resample": "1d"})],
            [("execute_python_plot", {"code": "print(len(df), list(df.columns)[:2])"})],
            [("execute_python_plot", {"code": PLOT_FUEL})],
        ],
    ),
    (
        "How much history do you see?",
        [[("search_shift_history", lambda m: {"query": f"history ctx={context_size(m)}"})]],
    ),
    (
        "Fetch the live data of 20 October 06:00 to 10:00 UTC.",
        [
            [
                (
                    "fetch_online_data",
                    {"start_time_utc": "2026-10-20T06:00:00Z", "end_time_utc": "2026-10-20T10:00:00Z"},
                )
            ]
        ],
    ),
    (
        "Keep searching the shift history.",
        [[("search_shift_history", {"query": f"round {i}"})] for i in range(9)],  # beyond the 8 tool rounds
    ),
]


def _chunks(text: str, size: int = 5) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


class ScriptedLLM:
    """Chat-model double following ``REFERENCE`` (plain and streamed calls give the same decisions)."""

    model = "fake/scripted"

    def __init__(self, script: list[tuple[str, list[Round] | str]] | None = None) -> None:
        self.script = dict(script or REFERENCE)
        self.calls: list[list[dict[str, Any]]] = []
        self.streamed = 0
        self.last_error: str | None = None

    # --- the decision
    def _decide(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
        self.calls.append([dict(m) for m in messages])
        last_user = max(
            i for i, m in enumerate(messages) if m.get("role") == "user" and isinstance(m.get("content"), str)
        )
        question = messages[last_user]["content"]
        rounds = self.script.get(question, [])
        if rounds == FAIL:
            raise RuntimeError("provider busy (429)")
        self.last_error = None  # as OpenRouterClient records a completed call
        done = sum(1 for m in messages[last_user:] if m.get("role") == "assistant" and m.get("tool_calls"))
        if done < len(rounds):
            calls = []
            for i, (name, args) in enumerate(rounds[done]):
                if isinstance(args, str):
                    raw = args
                else:
                    raw = json.dumps(args(messages) if callable(args) else args)
                q = list(self.script).index(question)
                calls.append(
                    {"id": f"call_{q}_{done}_{i}", "type": "function", "function": {"name": name, "arguments": raw}}
                )
            return None, calls
        tools = [m for m in messages[last_user:] if m.get("role") == "tool"]
        digest = normalise([str(t.get("content"))[:60] for t in tools])
        text = f"<think>check {len(tools)} results</think>Answer to '{question}': {len(tools)} tool results {digest}"
        return text, []

    # --- plain call (page 7)
    def chat_completions(self, *, messages: list[dict[str, Any]], tools: Any = None, tool_choice: Any = "auto") -> Any:
        text, calls = self._decide(messages)
        message = SimpleNamespace(content=text, tool_calls=[_sdk_call(c) for c in calls] or None)
        return SimpleNamespace(model=self.model, choices=[SimpleNamespace(message=message)])

    # --- streamed call (agent service)
    def chat_completions_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: Any = None,
        tool_choice: Any = "auto",
        on_token: Callable[[str], None] | None = None,
    ) -> Any:
        self.streamed += 1
        text, calls = self._decide(messages)
        chunks: list[dict[str, Any]] = []
        for piece in _chunks(text or ""):
            if piece:
                chunks.append({"model": self.model, "choices": [{"delta": {"content": piece}}]})
        for index, call in enumerate(calls):
            arguments = call["function"]["arguments"]
            half = len(arguments) // 2
            chunks.append(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": call["id"],
                                        "function": {"name": call["function"]["name"], "arguments": arguments[:half]},
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
            chunks.append(
                {
                    "choices": [
                        {"delta": {"tool_calls": [{"index": index, "function": {"arguments": arguments[half:]}}]}}
                    ]
                }
            )
        return assemble_stream(chunks, on_token=on_token, default_model=self.model)

    # --- what OpenRouterClient offers besides the calls
    def record_failure(self, exc: Exception) -> None:
        self.last_error = f"{type(exc).__name__}: {exc}"

    def unavailable_message(self) -> str:
        return "Medium effort level models are busy. Please try another mode."

    def usage_metadata(self) -> dict[str, Any]:
        failed = self.last_error is not None
        return {
            "reasoning_level": "Medium",
            "actual_model": None if failed else self.model,
            "model_status": "failed" if failed else "completed",
            "model_error": self.last_error,
        }


def _sdk_call(call: dict[str, Any]) -> Any:
    """A tool call as the OpenAI SDK returns it (attributes, not keys)."""
    function = SimpleNamespace(name=call["function"]["name"], arguments=call["function"]["arguments"])
    return SimpleNamespace(id=call["id"], type="function", function=function)
