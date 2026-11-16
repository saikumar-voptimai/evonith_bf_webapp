"""FurnaceMind's plant skills (user decision 13 Nov 2026): each skill computes its numbers from the plant's own records
and hands the model a filled form; nothing names a V-OptimAIse method or model; a skill whose data is missing says so
instead of raising. Synthetic data: ``tests/plant_skill_fakes.py``."""

from __future__ import annotations

import pandas as pd
import pytest

from agents.furnacemind import plant_skills, tool_state
from agents.furnacemind import skill_stats as st
from agents.furnacemind.plant_facts import config
from agents.furnacemind.skill_guard import method_words_in
from tests.plant_skill_fakes import NOW, FailingPlantData, FakePlantData

SHIFT = ("2026-11-13", "B")  # 14:00-22:00 IST; NOW is 21:00, so the shift is running
ARGS = {"shift_pulse": SHIFT}


def run(name: str, data=None) -> plant_skills.SkillOutput:
    return plant_skills.SKILLS[name](data or FakePlantData(), config(), *ARGS.get(name, ()))


def section(prompt: str, heading: str) -> str:
    """The data section of ``prompt`` that starts with ``heading`` (a line), up to the next blank line."""
    newline = chr(10)
    start = prompt.index(newline + heading) + 1
    end = prompt.find(newline * 2, start)
    return prompt[start:] if end < 0 else prompt[start:end]


# ---------------------------------------------------------------------------------------------------- statistics
def test_usual_value_and_spread_resist_the_odd_bad_reading() -> None:
    center, spread = st.center_spread([10, 11, 9, 10, 10, 11, 9, 500])
    assert center == pytest.approx(10.0)
    assert spread == pytest.approx(1.4826)  # the 500 does not widen it
    assert st.center_spread([5, 5, 5]) == (5.0, None)
    assert st.center_spread([]) == (None, None)


def test_a_reading_states_its_distance_direction_and_materiality() -> None:
    q = {
        "fuel_rate": {"label": "Fuel rate", "unit": "kg/tHM", "digits": 1, "better": "lower"},
        "flow": {"label": "Flow", "unit": "m³/h", "digits": 1, "better": "none", "floor_pct": 3.0},
    }
    worse = st.compare("fuel_rate", [570.0, 572.0], [560, 561, 559, 560, 562, 558], q)
    assert worse.state(2.0, 3.0).startswith("far outside, higher") and worse.direction() == "worse"
    steady = st.compare("flow", 187.0, [189.0, 189.1, 188.9, 189.0, 189.05], q)
    assert abs(steady.spreads) > 3 and steady.state(2.0, 3.0) == "usual"  # 1 % on a steady flow: below the floor


def test_cleaning_drops_readings_outside_their_valid_range() -> None:
    df = pd.DataFrame({"coal_rate_actual_value": [195.0, 428.0, 0.0, 201.0], "other": [1, 2, 3, 4]})
    out = st.clean(df, {"coal_rate_actual_value": {"valid": [80, 320]}})
    assert out["coal_rate_actual_value"].isna().tolist() == [False, True, True, False]
    assert out["other"].tolist() == [1, 2, 3, 4]


def test_shift_starts_and_trends() -> None:
    idx = pd.DatetimeIndex(["2026-11-13 05:59", "2026-11-13 06:00", "2026-11-13 21:59", "2026-11-13 22:00"])
    assert [str(t) for t in st.shift_start(idx)] == [
        "2026-11-12 22:00:00",
        "2026-11-13 06:00:00",
        "2026-11-13 14:00:00",
        "2026-11-13 22:00:00",
    ]
    assert st.shift_name(pd.Timestamp("2026-11-13 14:00")) == "13 Nov B"
    hours = pd.date_range("2026-11-13 00:00", periods=6, freq="1h")
    assert st.slope_per_hour(pd.Series([1480, 1478, 1476, 1474, 1472, 1470.0], index=hours)) == pytest.approx(-2.0)


def test_formatting_uses_units_no_break_spaces_and_true_minus() -> None:
    assert st.fmt(1234.56, 1, "kg/tHM") == "1,234.6 kg/tHM"
    assert st.fmt_signed(-3.0, 1, "%") == "−3.0 %"
    assert st.fmt(None) == "—" and st.fmt(float("nan")) == "—"


# --------------------------------------------------------------------------------------------------------- skills
@pytest.mark.parametrize("name", list(plant_skills.SKILLS))
def test_every_skill_fills_its_form_with_a_chart_and_no_method_word(name: str) -> None:
    out = run(name)
    assert out.prompt.startswith(f"SKILL: {plant_skills.TITLES[name]}")
    assert out.chart is not None and out.chart_id == name
    assert f'show_skill_chart(chart_id="{name}")' in out.prompt
    assert method_words_in(out.prompt) == []
    assert "DATA NOT AVAILABLE" not in out.prompt
    title = out.chart.layout.title.text or ""
    assert method_words_in(title) == []


@pytest.mark.parametrize("name", list(plant_skills.SKILLS))
def test_a_skill_without_data_says_what_is_missing_and_never_raises(name: str) -> None:
    out = run(name, FailingPlantData())
    assert method_words_in(out.prompt) == []
    assert "DATA NOT AVAILABLE (say so in the note; never fill these in)" in out.prompt
    assert "unreachable" in out.prompt


def test_shift_pulse_flags_the_movers_and_the_running_shift() -> None:
    prompt = run("shift_pulse").prompt
    assert "13 Nov B, running, first 7.0 h of 8" in prompt
    movers = section(prompt, "MOVERS (")
    assert "Fuel rate" in movers and "Coke rate" in movers
    readings = section(prompt, "READINGS (")
    pci = next(line for line in readings.splitlines() if line.startswith("| PCI rate"))
    assert "428" not in pci  # the stuck reading was dropped


def test_thermal_watch_reads_the_cold_taps_against_the_plant_range() -> None:
    prompt = run("thermal_watch").prompt
    assert "plant range 1440–1520 °C" in prompt
    assert "below" in section(prompt, "LAST TAPS")
    assert "COKE-04" in prompt  # sized with the plant fact register


def test_gas_flow_lens_finds_the_lower_stack_spread() -> None:
    prompt = run("gas_flow_lens").prompt
    spreads = section(prompt, "SPREADS ACROSS THE SECTORS")
    lower = next(line for line in spreads.splitlines() if line.startswith("| Lower stack"))
    assert "unusual" in lower or "far outside" in lower
    swings = section(prompt, "SWINGS:")
    assert "far outside" in next(line for line in swings.splitlines() if "Hot blast pressure swing" in line)


def test_wall_heat_map_names_the_risen_cell_and_stave_and_keeps_steady_water_usual() -> None:
    prompt = run("wall_heat_map").prompt
    rises = section(prompt, "RISES (")
    assert "R9 Q3" in rises
    assert "R9 stave 23/24" in section(prompt, "STAVES (")
    water = section(prompt, "COOLING WATER:")
    assert "| usual |" in next(line for line in water.splitlines() if line.startswith("| Bottom flow"))


def test_slag_burden_check_sees_the_ore_share_rise() -> None:
    prompt = run("slag_burden_check").prompt
    blend = section(prompt, "CHARGED BLEND")
    ore = next(line for line in blend.splitlines() if line.startswith("| Lump ore share"))
    assert "higher" in ore
    assert "plant band 1.005–1.155" in prompt


def test_fuel_drivers_splits_the_gap_and_sizes_it_with_the_plant_facts() -> None:
    prompt = run("fuel_drivers").prompt
    assert "FUEL GAP" in prompt and "COKE-05" in prompt
    assert "The plant facts account for" in prompt


def test_trial_board_reads_the_registers_trial_list_with_the_facts_and_why() -> None:
    prompt = run("trial_board").prompt
    trials = section(prompt, "TRIAL LIST")
    rows = [line for line in trials.splitlines() if line.startswith("| ") and line[2:3].isdigit()]
    assert [r.split("|")[2].split()[0] for r in rows] == [
        "T1",
        "T2",
        "T3",
        "T5",
        "T6",
        "T7",
        "T8",
        "TR-2",
        "TR-3",
        "TR-1b",
    ]  # rank order of the register (slide 46)
    assert "Saves Rs 185–197/tHM" in rows[0] and "COKE-05, GAS-01" in rows[0]
    facts = section(prompt, "FACTS THE TRIALS WOULD SETTLE")
    assert "grade C: Published value from other furnaces" in facts
    assert "Q6" in section(prompt, "OTHER TESTS THE FACTS NAME")
    assert "four-week trial arm" in section(prompt, "TRIAL RULES")
    assert "COKE-01 (PCI rate)" in section(prompt, "LEVER RANGES")


# ---------------------------------------------------------------------------------------------- run and the chart
def test_run_stores_the_chart_and_show_skill_chart_shows_it(monkeypatch) -> None:
    import agents.furnace_tools as ft

    monkeypatch.setattr(plant_skills, "PlantData", FakePlantData)
    state: dict = {}
    with tool_state.bind(state):
        prompt = plant_skills.run("wall_heat_map")
        assert "wall_heat_map" in state[plant_skills.CHARTS]
        message = ft.execute_openai_tool_call(name="show_skill_chart", arguments={"chart_id": "wall_heat_map"})
        missing = ft.execute_openai_tool_call(name="show_skill_chart", arguments={"chart_id": "nope"})
    assert prompt.startswith("SKILL: Wall Heat Map")
    assert message.startswith("Showing the skill chart") and state["fm_fig"] is not None
    assert "No chart 'nope'" in missing


def test_a_skill_that_raises_returns_a_prompt_that_forbids_guessing(monkeypatch) -> None:
    def broken(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setitem(plant_skills.SKILLS, "fuel_drivers", broken)
    with tool_state.bind({}):
        prompt = plant_skills.run("fuel_drivers", data=FakePlantData())
    assert "could not compute its numbers" in prompt and "Do not estimate any number" in prompt


def test_the_page_shift_is_read_up_to_now_when_it_is_still_running() -> None:
    data = FakePlantData()
    out = plant_skills.shift_pulse(data, config(), *SHIFT)
    online = next(c for c in data.calls if c[0] == "online")
    assert online[1] == ("process_params",) and NOW < pd.Timestamp("2026-11-13 22:00")
    assert "(13 Nov 14:00 – 13 Nov 21:00 IST)" in out.prompt
