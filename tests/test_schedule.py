"""Schedule of the worker (SV-1.7, #287): slots in IST, the slot of a cron run, slots served by records, the catch-up
plan at worker start, the schedule audit and the runtime statistics per kind. Pure: records in, decisions out."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from services import jobstats
from services import schedule as sch
from services.jobs import CATALOG

M = timedelta(minutes=1)
H = timedelta(hours=1)
TWIN = CATALOG["twin.evaluate"]
STEP2 = CATALOG["optimiser.step2"]
SI = CATALOG["forecast.si"]


def ist(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=sch.IST).astimezone(UTC)


def rec(kind, slot=None, *, status="done", by="scheduler", submitted=None, **kw):
    submitted = submitted or (slot + timedelta(seconds=0.2))
    return {
        "id": uuid.uuid4().hex,
        "kind": kind.name if hasattr(kind, "name") else kind,
        "status": status,
        "submitted_by": by,
        "submitted_at": submitted,
        "scheduled_for": slot,
        **kw,
    }


# ------------------------------------------------------------------------------------------------------------ slots
def test_slots_are_the_ist_times_as_utc_in_half_open_periods() -> None:
    day = sch.slots(STEP2.schedule, ist("2026-10-27 00:00"), ist("2026-10-28 00:00"))
    assert day == [ist("2026-10-27 06:15"), ist("2026-10-27 14:15"), ist("2026-10-27 22:15")]
    assert day[0] == datetime(2026, 10, 27, 0, 45, tzinfo=UTC)
    assert sch.slots(STEP2.schedule, ist("2026-10-27 06:15"), ist("2026-10-27 14:15")) == [ist("2026-10-27 06:15")]
    assert sch.slots(STEP2.schedule, ist("2026-10-27 06:15"), ist("2026-10-27 06:15")) == []
    assert len(sch.slots(SI.schedule, ist("2026-10-27 00:00"), ist("2026-10-28 00:00"))) == 288
    assert len(sch.slots(TWIN.schedule, ist("2026-10-20 00:00"), ist("2026-10-27 00:00"))) == 7 * 24
    nightly = CATALOG["calibration.nightly"].schedule  # 02:30 IST is 21:00 UTC of the day before
    assert sch.next_slot(nightly, ist("2026-10-27 23:00")) == ist("2026-10-28 02:30")
    assert sch.next_slot(nightly, ist("2026-10-28 02:30")) == ist("2026-10-29 02:30")


def test_last_slot_interval_and_the_slot_of_a_cron_run() -> None:
    assert sch.last_slot(TWIN.schedule, ist("2026-10-27 11:09:59")) == ist("2026-10-27 10:10")
    assert sch.last_slot(TWIN.schedule, ist("2026-10-27 11:10")) == ist("2026-10-27 11:10")
    assert sch.interval(TWIN.schedule) == H and sch.interval(SI.schedule) == 5 * M
    assert sch.interval(STEP2.schedule) == 8 * H and sch.interval(CATALOG["learn.shadow"].schedule) == 24 * H
    # arq runs a cron entry at second 0.123456 of the slot (later when busy); it may enqueue it 1 s before.
    assert sch.slot_of(TWIN.schedule, ist("2026-10-27 11:10:00.123456")) == ist("2026-10-27 11:10")
    assert sch.slot_of(TWIN.schedule, ist("2026-10-27 11:12:40")) == ist("2026-10-27 11:10")
    assert sch.slot_of(TWIN.schedule, ist("2026-10-27 11:09:59")) == ist("2026-10-27 11:10")


def test_schedules_are_described_as_the_jobs_api_states_them() -> None:
    assert sch.describe(TWIN.schedule) == "every hour at :10"
    assert sch.describe(STEP2.schedule) == "06:15, 14:15, 22:15"
    assert sch.describe(SI.schedule) == "every 5 min" and sch.describe(None) is None


def test_the_scheduled_kinds_end_before_their_next_slot() -> None:
    """The tries of a scheduled job end before the next slot of its kind (retries fail at their data check, the last
    try may use its whole time limit), so a slot is not skipped because the previous job is still running."""
    for kind in CATALOG.values():
        if kind.schedule is None:
            continue
        span = (kind.max_tries - 1) * kind.retry_delay_s + kind.timeout_s
        assert span < sch.interval(kind.schedule).total_seconds(), kind.name
    assert (TWIN.max_tries - 1) * TWIN.retry_delay_s == 15 * 60  # tries at HH:10, :15, :20, :25


# ------------------------------------------------------------------------------------------------------- matching
def test_records_serve_their_slot_and_older_records_their_submission_slot() -> None:
    records = [
        rec(TWIN, ist("2026-10-27 10:10")),
        rec(TWIN, None, submitted=ist("2026-10-27 11:10:01")),  # written before scheduled_for existed
        rec(TWIN, None, submitted=ist("2026-10-27 12:45")),  # 35 min after its slot: beyond the catch-up window
        rec(TWIN, None, submitted=ist("2026-10-27 13:11"), by="sv1"),  # a user's run serves no slot
        rec(SI, ist("2026-10-27 10:10")),  # another kind
    ]
    served = sch.served(records, TWIN.name, TWIN.schedule)
    assert sorted(served) == [ist("2026-10-27 10:10"), ist("2026-10-27 11:10")]
    legacy_si = [rec(SI, None, submitted=ist("2026-10-27 10:06"))]  # 6 min late for a 5-min schedule: slot 10:05
    assert list(sch.served(legacy_si, SI.name, SI.schedule)) == [ist("2026-10-27 10:05")]
    ran = rec(TWIN, ist("2026-10-27 10:10"), status="running")  # a run, then a skipped record of the same slot
    skip = rec(TWIN, ist("2026-10-27 10:10"), status="cancelled", by=sch.SKIPPED_BY, submitted=ist("2026-10-27 10:11"))
    assert sch.served([ran, skip], TWIN.name, TWIN.schedule)[ist("2026-10-27 10:10")] is ran
    assert (
        sch.served([skip, ran | {"submitted_at": ist("2026-10-27 10:12")}], TWIN.name, TWIN.schedule)[
            ist("2026-10-27 10:10")
        ]["status"]
        == "running"
    )


# ------------------------------------------------------------------------------------------------------- catch-up
def plan_of(*args, **kw):
    return [(d.kind, d.slot, d.action) for d in sch.catch_up_plan(*args, **kw)]


def test_the_latest_slot_runs_at_start_when_missed_by_less_than_30_min() -> None:
    kinds = [TWIN, STEP2]
    records = [rec(TWIN, ist("2026-10-27 10:10")), rec(STEP2, ist("2026-10-27 06:15"))]
    since = ist("2026-10-27 11:02")  # last heartbeat of the stopped worker
    assert plan_of(kinds, records, ist("2026-10-27 11:25"), since=since) == [
        ("twin.evaluate", ist("2026-10-27 11:10"), "submit")
    ]
    late = sch.catch_up_plan(kinds, records, ist("2026-10-27 11:41"), since=since)
    assert [(d.kind, d.action) for d in late] == [("twin.evaluate", "skip")]
    assert "11:10" in late[0].reason and "31 min later" in late[0].reason
    exactly = plan_of(kinds, records, ist("2026-10-27 11:40"), since=since)
    assert exactly == [("twin.evaluate", ist("2026-10-27 11:10"), "submit")]  # 30 min: still run


def test_every_slot_of_an_outage_is_recorded_and_only_the_latest_runs() -> None:
    since, now = ist("2026-10-27 06:12"), ist("2026-10-27 10:20")
    records = [rec(TWIN, ist("2026-10-27 06:10"))]
    plan = plan_of([TWIN], records, now, since=since)
    assert plan[0] == ("twin.evaluate", ist("2026-10-27 10:10"), "submit")
    assert [p[1] for p in plan[1:]] == [ist(f"2026-10-27 0{h}:10") for h in (9, 8, 7)]
    assert {p[2] for p in plan[1:]} == {"skip"}
    si = sch.catch_up_plan([SI], [], ist("2026-10-27 10:22"), since=ist("2026-10-27 10:08"))
    assert [(d.slot, d.action) for d in si] == [(ist("2026-10-27 10:20"), "submit"), (ist("2026-10-27 10:15"), "skip"),
                                                (ist("2026-10-27 10:10"), "skip")]  # fmt: skip
    assert "only the latest slot of a kind, 27 Oct 10:20 IST, runs at start" in si[1].reason
    capped = sch.catch_up_plan([SI], [], ist("2026-10-28 10:02"), since=ist("2026-10-27 10:00"), max_skipped=10)
    assert [d.action for d in capped] == ["submit"] + ["skip"] * 10  # 288 slots missed: the 10 newest recorded
    assert capped[1].slot == ist("2026-10-28 09:55") and capped[0].slot == ist("2026-10-28 10:00")


def test_a_slot_older_than_a_served_one_is_not_run_late() -> None:
    records = [rec(SI, ist("2026-10-27 10:05"))]
    plan = plan_of([SI], records, ist("2026-10-27 10:07"), since=ist("2026-10-27 09:59"))
    assert plan == [("forecast.si", ist("2026-10-27 10:00"), "skip")]


def test_slots_up_to_one_heartbeat_before_the_last_one_are_checked() -> None:
    """The worker may stop up to one heartbeat interval after its last mark, before the cron entry of a slot ran."""
    plan = plan_of([TWIN], [], ist("2026-10-27 11:20"), since=ist("2026-10-27 11:10:08"))
    assert plan == [("twin.evaluate", ist("2026-10-27 11:10"), "submit")]


def test_a_first_start_runs_only_a_recent_latest_slot() -> None:
    kinds = [TWIN, STEP2]
    assert plan_of(kinds, [], ist("2026-10-27 11:25"), since=None) == [
        ("twin.evaluate", ist("2026-10-27 11:10"), "submit")
    ]
    assert plan_of(kinds, [], ist("2026-10-27 11:45"), since=None) == []  # nothing is recorded without a heartbeat
    served = [rec(TWIN, ist("2026-10-27 11:10"))]
    assert plan_of(kinds, served, ist("2026-10-27 11:25"), since=None) == []


def test_a_skipped_record_serves_its_slot_at_the_next_start() -> None:
    since, now = ist("2026-10-27 06:12"), ist("2026-10-27 08:50")
    first = sch.catch_up_plan([TWIN], [], now, since=since)
    records = [rec(TWIN, d.slot, status="cancelled", by=sch.SKIPPED_BY) for d in first]
    assert sch.catch_up_plan([TWIN], records, now + M, since=now) == []


# ---------------------------------------------------------------------------------------------------------- audit
def test_the_audit_gives_each_slot_its_state() -> None:
    start, end = ist("2026-10-27 10:00"), ist("2026-10-27 16:00")
    records = [
        rec(TWIN, ist("2026-10-27 10:10")),
        rec(TWIN, ist("2026-10-27 11:10"), status="failed", error="RuntimeError: balance did not solve"),
        rec(TWIN, ist("2026-10-27 12:10"), status="cancelled", by=sch.SKIPPED_BY, error="Skipped: worker down"),
        rec(TWIN, None, submitted=ist("2026-10-27 13:12"), by="sv1"),  # 13:10 has no scheduler record: missed
        rec(TWIN, ist("2026-10-27 14:10"), status="running"),
        rec(TWIN, None, submitted=ist("2026-10-27 15:10:02")),  # before twin_0002: matched by submission
    ]
    res = sch.audit(records, [TWIN, STEP2], start, end)
    a = res["twin.evaluate"]
    assert a.expected == 6 and a.schedule == "every hour at :10"
    assert a.counts == {"done": 2, "failed": 1, "cancelled": 0, "running": 1, "skipped": 1, "missed": 1}
    assert [(p["slot"], p["state"]) for p in a.problems] == [
        (ist("2026-10-27 11:10"), "failed"),
        (ist("2026-10-27 12:10"), "skipped"),
        (ist("2026-10-27 13:10"), "missed"),
        (ist("2026-10-27 14:10"), "running"),
    ]
    assert a.problems[1]["reason"] == "Skipped: worker down" and not a.clean
    step2 = res["optimiser.step2"]
    assert step2.expected == 1 and step2.counts["missed"] == 1  # 14:15
    clean = sch.audit([rec(TWIN, ist("2026-10-27 10:10"))], [TWIN], start, ist("2026-10-27 11:00"))
    assert clean["twin.evaluate"].clean and clean["twin.evaluate"].as_dict()["done"] == 1


def test_the_audit_leaves_out_slots_of_the_last_minute() -> None:
    res = sch.audit([], [TWIN], ist("2026-10-27 10:00"), ist("2026-10-27 12:00"), now=ist("2026-10-27 11:10:30"))
    assert res["twin.evaluate"].expected == 1  # 10:10; the 11:10 job may not be recorded yet


def test_fourteen_days_of_served_slots_are_clean() -> None:
    kinds = [k for k in CATALOG.values() if k.schedule is not None]
    start, end = ist("2026-10-13 00:00"), ist("2026-10-27 00:00")
    records = [rec(k, t) for k in kinds for t in sch.slots(k.schedule, start, end)]
    res = sch.audit(records, kinds, start, end)
    assert all(a.clean and a.counts["done"] == a.expected for a in res.values())
    assert res["forecast.si"].expected == 14 * 288 and res["system.housekeeping"].expected == 14


# ------------------------------------------------------------------------------------------------ runtime statistics
def test_runtime_statistics_use_the_execution_time_over_the_tries() -> None:
    t0 = ist("2026-10-27 10:10")
    runs = [5.0 + i for i in range(20)]
    records = [
        rec(TWIN, t0 + H * i, started_at=t0 + H * i + timedelta(seconds=2), finished_at=t0 + H * i + 900 * M / 60,
            run_s=r)
        for i, r in enumerate(runs)
    ]  # fmt: skip
    records += [
        rec(TWIN, t0 + 20 * H, status="failed", started_at=t0 + 20 * H, finished_at=t0 + 20 * H + M, run_s=60.0),
        rec(TWIN, t0 + 21 * H, status="cancelled", by=sch.SKIPPED_BY, finished_at=t0 + 21 * H),
        rec(TWIN, None, submitted=t0 + 22 * H, by="sv1", started_at=t0 + 22 * H, finished_at=t0 + 22 * H + 30 * M / 60),
    ]
    st = jobstats.stats(records, [TWIN, STEP2], start=t0, end=t0 + 24 * H)
    a = st["twin.evaluate"]
    assert (a.runs, a.done, a.failed, a.skipped) == (22, 21, 1, 1)
    done = [*runs, 30.0]  # the user's run has no run_s: finish minus start, 30 s
    assert a.runtime_p50_s == pytest.approx(np.percentile(done, 50))
    assert a.runtime_p95_s == pytest.approx(np.percentile(done, 95)) and a.runtime_max_s == 30.0
    assert a.start_delay_p95_s == pytest.approx(2.0)  # from the slot; the user's run from its submission (0 s)
    assert a.last_status == "done" and a.last_finished_at == t0 + 22 * H + 30 * M / 60
    assert st["optimiser.step2"].runs == 0 and st["optimiser.step2"].runtime_p95_s is None
    doc = jobstats.document(st, start=t0, end=t0 + 24 * H, source="redis")
    assert doc["kinds"][0]["kind"] == "twin.evaluate" and isinstance(doc["kinds"][0]["last_finished_at"], str)
