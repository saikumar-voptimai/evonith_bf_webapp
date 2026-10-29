"""twin-api live routes: the event stream (#276) and the job routes (#277).

The stream tests run the application under uvicorn in a thread (the test client buffers whole responses, an event
stream never ends by itself) with the in-process event bus. The job tests use a throw-away redis-server.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import APIRouter, Depends, Request
from fastapi.testclient import TestClient

from api.auth import require_role
from api.live import live_state
from tests.redis_server import needs_redis, redis_server
from tests.test_api_auth import PASSWORDS, build

# ---------------------------------------------------------------- helpers


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def add_publish_probe(app) -> None:
    """``POST /api/_probe/publish``: publish an event on the process's bus (in the server's event loop)."""
    router = APIRouter(prefix="/api/_probe", tags=["probe"], dependencies=[Depends(require_role("admin"))])

    @router.post("/publish")
    async def publish(body: dict[str, Any], request: Request) -> dict[str, str]:
        event = await live_state(request).bus.publish(body["type"], body["data"])
        return {"id": event.id}

    app.include_router(router)


def session_cookie(app, username: str) -> str:
    """Sign in through the API and return the ``Cookie`` header value of the session."""
    client = TestClient(app, base_url="https://testserver")
    r = client.post("/api/auth/login", json={"username": username, "password": PASSWORDS[username][0]})
    assert r.status_code == 200, r.text
    return f"bf_session={client.cookies['bf_session']}"


@contextmanager
def live_server(app) -> Iterator[str]:
    port = _free_port()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", lifespan="on", timeout_graceful_shutdown=1
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(10)


def frames(response: httpx.Response) -> Iterator[dict[str, str]]:
    """Server-sent event frames of a streaming response, one dict per frame (``id``, ``event``, ``data``, ``retry``)."""
    frame: dict[str, str] = {}
    for line in response.iter_lines():
        if line == "":
            if frame:
                yield frame
            frame = {}
            continue
        name, _, value = line.partition(":")
        frame[name] = value.removeprefix(" ")


def next_frames(stream: Iterator[dict[str, str]], count: int, *, skip_heartbeats: bool = True) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    for frame in stream:
        if skip_heartbeats and frame.get("event") == "heartbeat":
            continue
        found.append(frame)
        if len(found) == count:
            return found
    raise AssertionError(f"stream ended after {len(found)} of {count} frames")


def progress(job_id: str, fraction: float) -> dict[str, Any]:
    return {
        "type": "job.progress",
        "data": {"job_id": job_id, "kind": "system.selftest", "status": "running", "progress": fraction},
    }


@pytest.fixture()
def stream_app():
    app = build(stream_heartbeat_s=0.3)
    add_publish_probe(app)
    return app


# ---------------------------------------------------------------- stream


def test_stream_needs_a_session(stream_app) -> None:
    with TestClient(stream_app, base_url="https://testserver") as client:
        r = client.get("/api/stream")
        assert r.status_code == 401
        assert r.headers["content-type"].startswith("application/problem+json")
        assert client.get("/api/stream/recent").status_code == 401


def test_stream_sends_published_events_with_their_ids(stream_app) -> None:
    cookie = session_cookie(stream_app, "adm")
    with live_server(stream_app) as base, httpx.Client(base_url=base, headers={"Cookie": cookie}, timeout=5) as http:
        with http.stream("GET", "/api/stream") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            assert response.headers["cache-control"] == "no-cache, no-transform"
            assert response.headers["x-accel-buffering"] == "no"
            stream = frames(response)
            assert next(stream) == {"retry": "3000"}
            ids = [http.post("/api/_probe/publish", json=progress("j1", f)).json()["id"] for f in (0.25, 0.5)]
            got = next_frames(stream, 2)
        assert [f["id"] for f in got] == ids
        assert [f["event"] for f in got] == ["job.progress", "job.progress"]
        payloads = [json.loads(f["data"]) for f in got]
        assert [p["data"]["progress"] for p in payloads] == [0.25, 0.5]
        assert all(p["id"] == f["id"] for p, f in zip(payloads, got, strict=True))


def test_stream_replays_missed_events_after_a_reconnect(stream_app) -> None:
    cookie = session_cookie(stream_app, "adm")
    with live_server(stream_app) as base, httpx.Client(base_url=base, headers={"Cookie": cookie}, timeout=5) as http:
        ids = [http.post("/api/_probe/publish", json=progress("j2", f)).json()["id"] for f in (0.1, 0.2, 0.3)]
        with http.stream("GET", "/api/stream", headers={"Last-Event-ID": ids[0]}) as response:
            stream = frames(response)
            assert next(stream) == {"retry": "3000"}
            replayed = next_frames(stream, 2)
            live_id = http.post("/api/_probe/publish", json=progress("j2", 0.4)).json()["id"]
            (live,) = next_frames(stream, 1)
        assert [f["id"] for f in replayed] == ids[1:]  # the events after Last-Event-ID, oldest first, no repeat
        assert live["id"] == live_id

        # A malformed Last-Event-ID is ignored: no replay, live events only.
        with http.stream("GET", "/api/stream", headers={"Last-Event-ID": "not-an-id"}) as response:
            stream = frames(response)
            assert next(stream) == {"retry": "3000"}
            fresh = http.post("/api/_probe/publish", json=progress("j2", 0.5)).json()["id"]
            assert next_frames(stream, 1)[0]["id"] == fresh

        # A new connection resumes with the last_event_id parameter; the header wins when both are sent.
        with http.stream("GET", "/api/stream", params={"last_event_id": ids[1]}) as response:
            stream = frames(response)
            assert next(stream) == {"retry": "3000"}
            assert [f["id"] for f in next_frames(stream, 3)] == [ids[2], live_id, fresh]
        with http.stream(
            "GET", "/api/stream", params={"last_event_id": ids[0]}, headers={"Last-Event-ID": live_id}
        ) as response:
            stream = frames(response)
            assert next(stream) == {"retry": "3000"}
            assert next_frames(stream, 1)[0]["id"] == fresh

        recent = http.get("/api/stream/recent", params={"limit": 2}).json()
        assert [e["id"] for e in recent] == [live_id, fresh]


def test_stream_heartbeats_and_ends_with_the_session(stream_app) -> None:
    cookie = session_cookie(stream_app, "op")
    with live_server(stream_app) as base, httpx.Client(base_url=base, headers={"Cookie": cookie}, timeout=5) as http:
        with http.stream("GET", "/api/stream") as response:
            stream = frames(response)
            assert next(stream) == {"retry": "3000"}
            beat = next(stream)
            assert beat["event"] == "heartbeat" and "id" not in beat  # heartbeats do not move Last-Event-ID
            assert json.loads(beat["data"])["type"] == "heartbeat"
            assert http.post("/api/auth/logout").status_code == 204
            started = time.monotonic()
            rest = list(stream)  # the server closes the stream at the next heartbeat check
        assert time.monotonic() - started < 3
        assert all(f.get("event") == "heartbeat" for f in rest)
        assert http.get("/api/stream").status_code == 401


def test_stream_ends_with_the_session_while_events_keep_arriving(stream_app) -> None:
    """Events more frequent than the heartbeat (worker status every 10 s against 15 s) must not postpone the
    session check."""
    cookie = session_cookie(stream_app, "adm")
    publisher_cookie = session_cookie(stream_app, "adm")  # a second session, still valid after the sign-out
    with live_server(stream_app) as base, httpx.Client(base_url=base, headers={"Cookie": cookie}, timeout=5) as http:
        stop = threading.Event()

        def publish() -> None:
            with httpx.Client(base_url=base, headers={"Cookie": publisher_cookie}, timeout=5) as publisher:
                n = 0
                while not stop.is_set():
                    publisher.post("/api/_probe/publish", json=progress("j3", (n % 10) / 10))
                    n += 1
                    stop.wait(0.1)

        thread = threading.Thread(target=publish, daemon=True)
        with http.stream("GET", "/api/stream") as response:
            stream = frames(response)
            assert next(stream) == {"retry": "3000"}
            thread.start()
            next_frames(stream, 3)
            # Revoke the streaming session; the publisher's session keeps the events coming.
            with httpx.Client(base_url=base, headers={"Cookie": cookie}, timeout=5) as other:
                assert other.post("/api/auth/logout").status_code == 204
            started = time.monotonic()
            rest: list[dict[str, str]] = []
            ended = False
            for frame in stream:
                rest.append(frame)
                if time.monotonic() - started > 3:
                    break
            else:
                ended = True
        stop.set()
        thread.join(5)
        assert ended, f"stream still open 3 s after the sign-out ({len(rest)} events since)"
        assert time.monotonic() - started < 2  # one check interval (0.3 s) plus margin
        assert all(f.get("event") == "job.progress" for f in rest)


# ---------------------------------------------------------------- jobs


@pytest.fixture()
def jobs_app(tmp_path):
    with redis_server(tmp_path) as url:
        yield build(redis_url=url)


class Users:
    """Requests as different users through one client: the client that ran the lifespan owns the event loop of
    the Redis connection, so every request that reaches Redis must go through it."""

    def __init__(self, app, client: TestClient) -> None:
        self.client = client
        self.cookies = {u: session_cookie(app, u) for u in PASSWORDS}

    def get(self, user: str, url: str, **kw: Any) -> httpx.Response:
        return self.client.get(url, headers={"Cookie": self.cookies[user]}, **kw)

    def post(self, user: str, url: str, **kw: Any) -> httpx.Response:
        return self.client.post(url, headers={"Cookie": self.cookies[user]}, **kw)


@needs_redis
def test_job_kinds_list_roles_schedules_and_parameter_schemas(jobs_app) -> None:
    with TestClient(jobs_app, base_url="https://testserver") as client:
        kinds = {k["name"]: k for k in Users(jobs_app, client).get("op", "/api/jobs/kinds").json()}
    selftest = kinds["system.selftest"]
    assert selftest["role"] == "admin" and selftest["timeout_s"] == 120
    assert set(selftest["params_schema"]["properties"]) == {"steps", "step_s", "cpu"}
    assert kinds["system.housekeeping"]["schedule"] == "04:00"


@needs_redis
def test_submit_checks_the_role_parameters_and_duplicates(jobs_app) -> None:
    with TestClient(jobs_app, base_url="https://testserver") as client:
        users = Users(jobs_app, client)
        selftest = {"kind": "system.selftest"}

        refused = users.post("sup", "/api/jobs", json=selftest)
        assert refused.status_code == 403 and "admin" in refused.json()["detail"]
        assert users.post("adm", "/api/jobs", json={"kind": "no.such"}).status_code == 404

        bad = users.post("adm", "/api/jobs", json={**selftest, "params": {"steps": 0}})
        assert bad.status_code == 422
        assert bad.json()["errors"][0]["loc"] == ["body", "params", "steps"]
        assert users.post("adm", "/api/jobs", json={**selftest, "params": {"speed": 1}}).status_code == 422

        first = users.post("adm", "/api/jobs", json={**selftest, "params": {"steps": 3}})
        assert first.status_code == 202
        job = first.json()["job"]
        assert first.json()["duplicate"] is False and job["status"] == "queued" and job["submitted_by"] == "adm"
        assert job["params"] == {"steps": 3, "step_s": 1.0, "cpu": True}  # defaults filled in before hashing

        again = users.post("adm", "/api/jobs", json={**selftest, "params": {"steps": 3, "cpu": True}})
        assert again.status_code == 202
        assert again.json() == {"job": job, "duplicate": True}
        other = users.post("adm", "/api/jobs", json={**selftest, "params": {"steps": 4}}).json()
        assert other["duplicate"] is False and other["job"]["id"] != job["id"]

        assert users.get("op", f"/api/jobs/{job['id']}").json() == job
        assert users.get("op", "/api/jobs/0123456789abcdef").status_code == 404
        listed = [j["id"] for j in users.get("op", "/api/jobs").json()]
        assert listed == [other["job"]["id"], job["id"]]  # newest first


@pytest.fixture()
def audit_store(tmp_path):
    """A SQLite twin store for the routes that write the audit log (a cancel, SV-4.8)."""
    from data.twin.engine import build_engine, session_factory
    from data.twin.models import TwinBase
    from data.twin.repositories import TwinStore
    from services import store as store_service

    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    store = TwinStore(session_factory(engine), engine)
    store_service.use(store)
    yield store
    store_service.reset()
    store.dispose()


@needs_redis
def test_cancel_ends_a_queued_job_and_publishes_job_done(jobs_app, audit_store) -> None:
    with TestClient(jobs_app, base_url="https://testserver") as client:
        users = Users(jobs_app, client)
        job = users.post("adm", "/api/jobs", json={"kind": "system.selftest"}).json()["job"]

        assert users.post("op", f"/api/jobs/{job['id']}/cancel").status_code == 403
        ended = users.post("sup", f"/api/jobs/{job['id']}/cancel")
        assert ended.status_code == 200
        record = ended.json()
        assert record["status"] == "cancelled" and record["cancel_requested"] is True
        assert record["error"] == "Cancelled before start."

        again = users.post("sup", f"/api/jobs/{job['id']}/cancel")
        assert again.status_code == 409 and "cancelled" in again.json()["detail"]
        assert [(e.action, e.actor) for e in audit_store.audit.query()] == [("job.cancel", "sup")]

        done = [e for e in users.get("op", "/api/stream/recent").json() if e["type"] == "job.done"]
        assert done[-1]["data"] == {
            "job_id": job["id"],
            "kind": "system.selftest",
            "status": "cancelled",
            "message": "Cancelled before start.",
            "duration_s": None,
        }

        # The hold on the input hash is released: the same request starts a new job.
        resubmitted = users.post("adm", "/api/jobs", json={"kind": "system.selftest"}).json()
        assert resubmitted["duplicate"] is False and resubmitted["job"]["id"] != job["id"]


def test_job_routes_answer_503_while_redis_is_unreachable() -> None:
    app = build(redis_url=f"redis://127.0.0.1:{_free_port()}/0")
    with TestClient(app, base_url="https://testserver") as client:
        users = Users(app, client)
        for r in (users.get("adm", "/api/jobs"), users.post("adm", "/api/jobs", json={"kind": "system.selftest"})):
            assert r.status_code == 503
            assert "Redis" in r.json()["detail"]
        assert users.get("adm", "/api/jobs/kinds").status_code == 200  # the catalogue needs no Redis
