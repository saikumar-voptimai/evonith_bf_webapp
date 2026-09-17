"""Role, ownership, and service-boundary tests for scheduled jobs."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from typing import Any

import pytest

import data.db as db_module
from data.db import ScheduledJobService
from utils.scheduled_job_access import (
    CREATE,
    DELETE_ALL,
    UPDATE_ALL,
    UPDATE_OWN,
    VIEW_ALL,
    ScheduledJobAuthorizationError,
    ScheduledJobPrincipal,
    can_create_job,
    can_delete_job,
    can_update_job,
    can_view_job,
)
from utils.scheduled_jobs import (
    LOCAL_TIMEZONE,
    archive_job,
    build_delivery_configuration,
    build_job_document,
    build_retry_configuration,
    build_schedule,
    clone_job_document,
    pause_job,
    request_run_now,
    resume_job,
)

NOW = LOCAL_TIMEZONE.localize(datetime(2026, 9, 17, 10, 0))
ROLE_PERMISSIONS = {
    "admin": {VIEW_ALL, CREATE, UPDATE_ALL, DELETE_ALL},
    "supervisor": {VIEW_ALL, CREATE, UPDATE_OWN},
    "user": {VIEW_ALL},
    "unknown": set(),
}


def _principal(role: str, username: str) -> ScheduledJobPrincipal:
    return ScheduledJobPrincipal.from_permissions(username, ROLE_PERMISSIONS[role])


def _job(owner: str, number: int, status: str = "active") -> dict[str, Any]:
    return build_job_document(
        job_name=f"Job {number}",
        instructions="Generate the scheduled operations report.",
        model_level="medium",
        status=status,
        schedule=build_schedule("Every Hour", execution_minute=30),
        delivery=build_delivery_configuration(),
        retry=build_retry_configuration(),
        created_by=owner,
        job_id=f"Job-{number}",
        now=NOW,
    )


class _Result:
    def __init__(self, *, rows=None, scalar_value=None, rowcount=1):
        self._rows = rows or []
        self._scalar = scalar_value
        self.rowcount = rowcount

    def mappings(self):
        return self

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._scalar


class _Engine:
    """Small parameter-aware SQLAlchemy engine fake for service policy tests."""

    def __init__(self, jobs: list[dict[str, Any]]):
        self.jobs = copy.deepcopy(jobs)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.next_job_number = 2001

    def begin(self):
        return self

    def connect(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, statement, params=None):
        sql, values = str(statement), dict(params or {})
        self.calls.append((sql, values))
        if "nextval('automation.scheduled_job_id_seq')" in sql:
            number = self.next_job_number
            self.next_job_number += 1
            return _Result(scalar_value=number)
        if "SELECT EXISTS" in sql:
            return _Result(scalar_value=False)
        if "SELECT job_description" in sql:
            jobs = self.jobs
            if job_id := values.get("job_id"):
                jobs = [job for job in jobs if job["job_id"] == job_id]
            return _Result(rows=[{"job_description": json.dumps(job)} for job in jobs])
        if "INSERT INTO automation.scheduled_jobs" in sql:
            self.jobs.append(json.loads(values["job_description"]))
            return _Result()
        if "UPDATE automation.scheduled_jobs" in sql:
            for index, job in enumerate(self.jobs):
                if job["job_id"] == values["job_id"]:
                    self.jobs[index] = json.loads(values["job_description"])
                    return _Result()
            return _Result(rowcount=0)
        if "DELETE FROM automation.scheduled_jobs" in sql:
            original_count = len(self.jobs)
            self.jobs = [job for job in self.jobs if job["job_id"] != values["job_id"]]
            return _Result(rowcount=original_count - len(self.jobs))
        return _Result()

    def dispose(self):
        pass


def _service(
    monkeypatch,
    jobs: list[dict[str, Any]],
    principal: ScheduledJobPrincipal | None,
) -> tuple[ScheduledJobService, _Engine]:
    engine = _Engine(jobs)
    monkeypatch.setattr(
        db_module, "build_relational_engine", lambda db_url=None: engine
    )
    monkeypatch.setattr(
        db_module,
        "build_relational_session_factory",
        lambda relational_engine: object(),
    )
    return (
        ScheduledJobService(db_url="postgresql://example", principal=principal),
        engine,
    )


@pytest.mark.parametrize("role", ["admin", "supervisor", "user"])
def test_all_supported_roles_can_view_and_list_every_job(
    monkeypatch, role: str
) -> None:
    jobs = [_job("sai", 1001), _job("sasi", 1002), _job("other", 1003)]
    principal = _principal(role, role)
    service, _ = _service(monkeypatch, jobs, principal)

    assert can_view_job(principal, jobs[0])
    assert [job["job_id"] for job in service.list_jobs()] == [
        "Job-1001",
        "Job-1002",
        "Job-1003",
    ]
    assert service.get_job("Job-1001") == jobs[0]


def test_unknown_role_is_default_deny(monkeypatch) -> None:
    principal = _principal("unknown", "mystery")
    service, _ = _service(monkeypatch, [_job("sai", 1001)], principal)

    assert not can_view_job(principal)
    assert not can_create_job(principal)
    assert not can_update_job(principal, _job("mystery", 1002))
    assert not can_delete_job(principal)
    with pytest.raises(ScheduledJobAuthorizationError, match="not authorized"):
        service.list_jobs()


@pytest.mark.parametrize(
    ("role", "username", "owner", "allowed"),
    [
        ("admin", "sai", "other", True),
        ("supervisor", "sasi", "sasi", True),
        ("supervisor", "sasi", "other", False),
        ("user", "john", "john", False),
    ],
)
def test_update_uses_persisted_ownership(
    monkeypatch, role: str, username: str, owner: str, allowed: bool
) -> None:
    persisted = _job(owner, 1001)
    replacement = copy.deepcopy(persisted)
    replacement["job_name"] = "Updated name"
    principal = _principal(role, username)
    service, engine = _service(monkeypatch, [persisted], principal)

    if allowed:
        service.update_job(persisted["job_id"], replacement)
        assert engine.jobs[0]["job_name"] == "Updated name"
    else:
        replacement["created_by"] = username  # Submitted ownership must not authorize.
        with pytest.raises(ScheduledJobAuthorizationError, match="not authorized"):
            service.update_job(persisted["job_id"], replacement)
        assert engine.jobs[0] == persisted


def test_owner_only_update_does_not_disclose_missing_jobs(monkeypatch) -> None:
    service, _ = _service(
        monkeypatch, [], _principal("supervisor", "sasi")
    )

    with pytest.raises(ScheduledJobAuthorizationError, match="not authorized"):
        service.update_job("Job-9999", _job("sasi", 9999))


def _action_document(action: str, job: dict[str, Any], actor: str) -> dict[str, Any]:
    if action == "pause":
        return pause_job(job, actor=actor, now=NOW)
    if action in {"activate", "resume"}:
        return resume_job(job, actor=actor, now=NOW)
    if action == "run":
        return request_run_now(job, actor=actor, now=NOW)
    return archive_job(job, actor=actor, now=NOW)


@pytest.mark.parametrize("action", ["activate", "pause", "resume", "run", "archive"])
@pytest.mark.parametrize(
    ("role", "username", "owner", "allowed"),
    [
        ("admin", "sai", "other", True),
        ("supervisor", "sasi", "sasi", True),
        ("supervisor", "sasi", "other", False),
        ("user", "john", "john", False),
    ],
)
def test_control_actions_share_update_authorization(
    monkeypatch,
    action: str,
    role: str,
    username: str,
    owner: str,
    allowed: bool,
) -> None:
    status = "draft" if action == "activate" else "active"
    persisted = _job(owner, 1001, status=status)
    if action == "resume":
        persisted = pause_job(persisted, actor=owner, now=NOW)
    replacement = _action_document(action, persisted, username)
    service, engine = _service(monkeypatch, [persisted], _principal(role, username))

    if allowed:
        service.update_job(persisted["job_id"], replacement)
        assert engine.jobs[0] == replacement
    else:
        with pytest.raises(ScheduledJobAuthorizationError, match="not authorized"):
            service.update_job(persisted["job_id"], replacement)
        assert engine.jobs[0] == persisted


@pytest.mark.parametrize(
    ("role", "username", "owner", "allowed"),
    [
        ("admin", "sai", "other", True),
        ("supervisor", "sasi", "sasi", False),
        ("supervisor", "sasi", "other", False),
        ("user", "john", "john", False),
    ],
)
def test_only_admin_can_delete(
    monkeypatch, role: str, username: str, owner: str, allowed: bool
) -> None:
    persisted = _job(owner, 1001)
    service, engine = _service(monkeypatch, [persisted], _principal(role, username))

    if allowed:
        service.delete_job(persisted["job_id"])
        assert engine.jobs == []
    else:
        with pytest.raises(ScheduledJobAuthorizationError, match="not authorized"):
            service.delete_job(persisted["job_id"])
        assert engine.jobs == [persisted]


@pytest.mark.parametrize(
    ("role", "username", "allowed"),
    [("admin", "sai", True), ("supervisor", "sasi", True), ("user", "john", False)],
)
def test_create_permission_and_authenticated_ownership(
    monkeypatch, role: str, username: str, allowed: bool
) -> None:
    submitted = _job("spoofed-owner", 1001)
    service, engine = _service(monkeypatch, [], _principal(role, username))

    if allowed:
        service.create_job(submitted)
        assert engine.jobs[0]["created_by"] == username
        assert submitted["created_by"] == "spoofed-owner"
    else:
        with pytest.raises(ScheduledJobAuthorizationError, match="not authorized"):
            service.create_job(submitted)
        assert engine.jobs == []


@pytest.mark.parametrize("field", ["job_id", "created_by", "created_at"])
def test_update_cannot_change_identity_or_ownership(monkeypatch, field: str) -> None:
    persisted = _job("owner", 1001)
    replacement = copy.deepcopy(persisted)
    replacement[field] = {
        "job_id": "Job-9999",
        "created_by": "new-owner",
        "created_at": "2026-09-18T10:00:00+05:30",
    }[field]
    service, _ = _service(monkeypatch, [persisted], _principal("admin", "admin"))

    with pytest.raises(ValueError, match="cannot be changed"):
        service.update_job(persisted["job_id"], replacement)


@pytest.mark.parametrize(
    ("role", "username", "owner", "allowed"),
    [
        ("admin", "sai", "other", True),
        ("supervisor", "sasi", "sasi", True),
        ("supervisor", "sasi", "other", False),
        ("user", "john", "john", False),
    ],
)
def test_clone_requires_update_access_to_persisted_source(
    monkeypatch, role: str, username: str, owner: str, allowed: bool
) -> None:
    source = _job(owner, 1001)
    clone = clone_job_document(
        source, created_by="spoofed-owner", job_id="Job-1002", now=NOW
    )
    service, engine = _service(monkeypatch, [source], _principal(role, username))

    if allowed:
        service.create_cloned_job(source["job_id"], clone)
        assert len(engine.jobs) == 2
        assert engine.jobs[1]["created_by"] == username
    else:
        with pytest.raises(ScheduledJobAuthorizationError, match="not authorized"):
            service.create_cloned_job(source["job_id"], clone)
        assert engine.jobs == [source]


@pytest.mark.parametrize("role", ["supervisor", "user"])
def test_histories_remain_globally_visible(monkeypatch, role: str) -> None:
    job = _job("another-user", 1001)
    for field in (
        "revision_history",
        "run_history",
        "delivery_history",
        "report_history",
        "error_history",
        "execution_requests",
        "action_history",
    ):
        job[field].append({"visible": field})
    service, _ = _service(monkeypatch, [job], _principal(role, role))

    visible = service.get_job(job["job_id"])
    assert visible is not None
    assert all(
        visible[field] == job[field] for field in job if field.endswith("history")
    )
    assert visible["execution_requests"] == job["execution_requests"]


def test_principal_free_service_retains_trusted_worker_access(monkeypatch) -> None:
    persisted = _job("owner", 1001)
    replacement = request_run_now(persisted, actor="worker", now=NOW)
    service, engine = _service(monkeypatch, [persisted], principal=None)

    service.update_job(persisted["job_id"], replacement)
    service.create_job(_job("system", 1002))
    assert engine.jobs[0] == replacement
    assert engine.jobs[1]["created_by"] == "system"
