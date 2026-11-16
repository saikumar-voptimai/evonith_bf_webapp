"""What a FurnaceMind quick skill may use and show (user decision 13 Nov 2026): the plant-data tools only, a system
prompt without the tool-routing notes and the live-twin routing, and the plant fact register without how its values
were worked out (no model or validation group, no basis, note or parameters, no fact whose text names a method)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agents.furnacemind import plant_facts
from agents.furnacemind.skill_guard import (
    METHOD_WORDS,
    SKILL_TOOLS,
    method_words_in,
    skill_persona,
    skill_system_prompt,
    skill_tools,
)

ROOT = Path(__file__).resolve().parents[1]
TWIN_TOOLS = {"get_twin_snapshot", "get_twin_series", "get_transit", "get_optimiser_result"}


def test_the_method_words_are_the_spas_list() -> None:
    """The Python list equals ``frontend/src/lib/copy/method-words.ts`` word for word, in order."""
    source = (ROOT / "frontend" / "src" / "lib" / "copy" / "method-words.ts").read_text(encoding="utf-8")
    block = source.split("export const METHOD_WORDS")[1].split("];")[0]
    spa = re.findall(r"\bw\('([^']+)'", block)
    assert spa == [m.word for m in METHOD_WORDS]


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("The calibration of the model", ["calibration"]),
        ("omega rose; ω = 0.82", ["omega", "ω"]),
        ("STRATA and FORGE against TRACE", ["STRATA", "FORGE", "TRACE"]),
        ("a two-zone Rist balance", ["two-zone", "Rist"]),
        ("trace the runner temperature; a strata of coke", []),
        ("Fuel rate 568.7 kg/tHM, 5× the usual spread", []),
    ],
)
def test_method_words_and_model_names_are_found(text: str, found: list[str]) -> None:
    assert method_words_in(text) == found


def test_a_skill_turn_offers_the_plant_data_tools_only() -> None:
    from agents.furnace_tools import get_openai_tool_schemas

    every = {t["function"]["name"] for t in get_openai_tool_schemas()}
    offered = {t["function"]["name"] for t in skill_tools(get_openai_tool_schemas())}
    assert TWIN_TOOLS <= every and not (offered & TWIN_TOOLS)
    assert offered <= SKILL_TOOLS
    assert {"show_skill_chart", "lookup_plant_facts", "fetch_online_data", "fetch_offline_data"} <= offered


def test_the_skill_system_prompt_names_no_model_and_routes_no_twin_tool() -> None:
    prompt = skill_system_prompt("Thermal Watch (database skill instruction):\nHearth thermal state.")
    assert method_words_in(prompt) == []
    for name in TWIN_TOOLS:
        assert name not in prompt
    assert "twin" not in skill_persona().lower()
    assert "Hearth thermal state." in prompt and "SKILL TURN RULES" in prompt


def test_the_service_builds_the_skill_prompt_on_a_skill_turn() -> None:
    from services import furnacemind as fm

    skill = fm._default_system_prompt(skill_id="thermal_watch", skill_context="Hearth thermal state.")
    assert skill == skill_system_prompt("Hearth thermal state.")


# ------------------------------------------------------------------------------------------------- fact register
def test_the_shown_register_hides_how_values_were_worked_out() -> None:
    facts = plant_facts.plant_facts()
    raw = json.loads((ROOT / "src" / "config" / "facts_register.json").read_text(encoding="utf-8"))["facts"]
    hidden_groups = {f["id"] for f in raw if f["group"] in ("MOD", "VAL")}
    retired = {f["id"] for f in raw if f["status"] == "Retired"}
    shown = {f["id"] for f in facts}
    assert facts and not (shown & hidden_groups) and not (shown & retired)
    for fact in facts:
        assert not {"basis", "note", "params", "se"} & set(fact), fact["id"]
        text = " ".join(
            str(fact.get(k) or "") for k in ("statement", "unit", "literature", "next_test", "validity_window")
        )
        assert method_words_in(text) == [], fact["id"]
    # Facts that name the models in their statement are left out entirely.
    assert "GAS-13" not in shown and "ASM-18" not in shown
    # The plant's bands and responses stay.
    assert {"BAND-02", "COKE-01", "FLOW-02"} <= shown


def test_open_facts_and_the_registers_trials() -> None:
    assert plant_facts.trial_codes("Q6, T5") == ["Q6", "T5"]
    assert plant_facts.trial_codes("Q12; production step trial") == ["Q12", "production step trial"]
    board = plant_facts.trials()
    listed = [code for code, t in board.items() if t["listed"]]
    assert listed == ["T1", "T2", "T3", "T5", "T6", "T7", "T8", "TR-2", "TR-3", "TR-1b"]
    assert {f["id"] for f in board["T5"]["facts"]} >= {"COKE-15", "GAS-06", "FLOW-02"}
    # The deck's words that name a model or method are replaced by the register's plant wording.
    assert board["TR-3"]["benefit"].startswith("Settles how much coke")
    assert board["T8"]["design"] == "Gradual, with a running watch on HMT and Si"
    assert board["T1"]["benefit"].startswith("Saves Rs 185–197/tHM")
    for code, group in board.items():
        assert method_words_in(json.dumps(group, ensure_ascii=False)) == [], code
        for fact in group["facts"]:
            assert not fact["id"].startswith(("MOD-", "VAL-")) and fact["id"] != "GAS-14"
            if not group["listed"]:
                assert fact["next_test"] and fact["why"]
                assert fact["grade"] in ("C", "D") or fact["status"] in (
                    "Assumed",
                    "Disputed",
                    "Measured gap",
                    "Instrument check needed",
                    "Modelled only",
                )


def test_the_lookup_tool_filters_and_never_shows_a_hidden_fact() -> None:
    from agents.furnace_tools import execute_openai_tool_call

    pci = json.loads(execute_openai_tool_call(name="lookup_plant_facts", arguments={"query": "PCI coke"}))
    assert any(f["id"] == "COKE-01" for f in pci["facts"]) and pci["register"]["name"]
    assert "basis" not in json.dumps(pci["facts"])
    hidden = json.loads(execute_openai_tool_call(name="lookup_plant_facts", arguments={"ids": ["GAS-13", "MOD-01"]}))
    assert hidden["facts"] == []
    trial = json.loads(execute_openai_tool_call(name="lookup_plant_facts", arguments={"trial": "T1"}))
    assert [t["id"] for t in trial["trials"]] == ["T1"] and trial["trials"][0]["change"] == "Hot blast 1,188 → 1,230 °C"
    assert {f["id"] for f in trial["facts"]} == {"COKE-05", "GAS-01"} and all("why" in f for f in trial["facts"])
    assert "detection" in trial["trial_rules"]
    assert method_words_in(json.dumps(trial, ensure_ascii=False)) == []
