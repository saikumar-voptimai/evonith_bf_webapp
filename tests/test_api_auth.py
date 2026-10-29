"""twin-api authentication and roles (#273): tokens, sign-in limit, sessions, origin guard, role x endpoint matrix."""

from __future__ import annotations

import base64
import json

import pytest
from fastapi import APIRouter, Depends
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.auth import PUBLIC_PATHS, require_role, required_role_of
from api.main import create_app
from api.security.limits import DenyList, FailureLimiter
from api.security.tokens import TokenError, decode, encode
from api.settings import ApiSettings
from services.auth import IdentityUnavailable, Principal
from services.session_tokens import verify_session

SECRET = "s" * 48
PASSWORDS = {"op": ("pw-op", "user"), "sup": ("pw-sup", "supervisor"), "adm": ("pw-adm", "admin")}


class FakeIdentity:
    def __init__(self) -> None:
        self.down = False
        self.calls = 0

    def authenticate(self, username: str, password: str) -> Principal | None:
        self.calls += 1
        if self.down:
            raise IdentityUnavailable("down")
        record = PASSWORDS.get(username)
        if record and record[0] == password:
            return Principal(username=username, role=record[1], user_id=f"id-{username}")
        return None


class Clock:
    def __init__(self, t: float = 1_793_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def build(**overrides):
    settings = ApiSettings(static_dir=None, jwt_secret=SECRET, **overrides)
    app = create_app(settings)
    state = app.state.auth
    state.identity = FakeIdentity()
    state.clock = Clock()
    app.state.limiter_clock = Clock(100.0)
    state.limiter = FailureLimiter(settings.login_max_failures, settings.login_window_s, clock=app.state.limiter_clock)
    state.denylist = DenyList(clock=state.clock)

    probe = APIRouter(prefix="/api/_probe", tags=["probe"])
    for role in ("user", "supervisor", "admin"):
        probe.add_api_route(
            f"/{role}", lambda: {"ok": True}, methods=["GET"], dependencies=[Depends(require_role(role))]
        )
    app.include_router(probe)
    return app


@pytest.fixture()
def app():
    return build()


def client_for(app) -> TestClient:
    # https base URL: the session cookie is Secure and only travels over https.
    return TestClient(app, base_url="https://testserver")


def with_token(app, token: str) -> TestClient:
    """A fresh client that presents ``token`` as its session cookie."""
    client = client_for(app)
    client.headers["Cookie"] = f"bf_session={token}"
    return client


def sign_in(app, username: str) -> TestClient:
    client = client_for(app)
    r = client.post("/api/auth/login", json={"username": username, "password": PASSWORDS[username][0]})
    assert r.status_code == 200, r.text
    return client


# ---------------------------------------------------------------- tokens


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def test_hs256_verifies_the_rfc_7515_example() -> None:
    """RFC 7515 appendix A.1: the published HS256 JWS verifies with its key; one changed byte does not."""
    key = base64.urlsafe_b64decode(
        "AyM1SysPpbyDfgZld3umj1qzKObwVMkoqQ-EstJQLr_T-1qS0gZH75aKtMN3Yj0iPS4hcgUuTwjAzZr1Z9CAow=="
    )
    header = "eyJ0eXAiOiJKV1QiLA0KICJhbGciOiJIUzI1NiJ9"
    payload = "eyJpc3MiOiJqb2UiLA0KICJleHAiOjEzMDA4MTkzODAsDQogImh0dHA6Ly9leGFtcGxlLmNvbS9pc19yb290Ijp0cnVlfQ"
    signature = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    claims = decode(f"{header}.{payload}.{signature}", key, now=1_300_819_000)
    assert claims["iss"] == "joe" and claims["http://example.com/is_root"] is True
    with pytest.raises(TokenError):
        decode(f"{header}.{payload}.e{signature[1:]}", key, now=1_300_819_000)


def test_token_round_trip_and_rejections() -> None:
    key = b"k" * 32
    token = encode({"sub": "op", "exp": 2000, "iat": 1000}, key)
    assert decode(token, key, now=1500)["sub"] == "op"

    header, payload, sig = token.split(".")
    forged = _b64url(json.dumps({"sub": "adm", "exp": 2000, "iat": 1000}).encode())
    none_header = _b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    hs512_header = _b64url(json.dumps({"alg": "HS512", "typ": "JWT"}).encode())
    cases = {
        "changed payload": f"{header}.{forged}.{sig}",
        "alg none": f"{none_header}.{payload}.",
        "other algorithm": f"{hs512_header}.{payload}.{sig}",
        "two segments": f"{header}.{payload}",
        "bad encoding": f"{header}.{payload}.!!",
    }
    for name, bad in cases.items():
        with pytest.raises(TokenError):
            decode(bad, key, now=1500)
        assert name
    with pytest.raises(TokenError):
        decode(token, b"x" * 32, now=1500)  # other key
    with pytest.raises(TokenError):
        decode(token, key, now=2031)  # expired beyond the 30 s leeway
    with pytest.raises(TokenError):
        decode(encode({"sub": "op", "exp": 9000, "iat": 5000}, key), key, now=1500)  # issued in the future
    with pytest.raises(TokenError):
        decode(encode({"sub": "op"}, key), key, now=1500)  # no expiry


# ---------------------------------------------------------------- sign-in and session


def test_sign_in_sets_a_strict_http_only_cookie(app) -> None:
    client = client_for(app)
    r = client.post("/api/auth/login", json={"username": "sup", "password": "pw-sup"})
    assert r.status_code == 200
    assert r.json()["user"] == {"username": "sup", "role": "supervisor", "user_id": "id-sup"}
    cookie = r.headers["set-cookie"].lower()
    # path=/ (not /api): the Streamlit pages under /legacy sign in from the same cookie (SV-6.6, #320)
    for part in ("bf_session=", "httponly", "secure", "samesite=strict", "path=/;", "max-age=43200"):
        assert part in cookie, part
    assert "path=/api" not in cookie
    me = client.get("/api/auth/me")
    assert me.status_code == 200 and me.json()["user"]["role"] == "supervisor"


def test_api_cookie_is_accepted_by_the_streamlit_rules(app) -> None:
    """The cookie the SPA holds signs the Streamlit session in: same key, same rules (SV-6.6, #320)."""
    state = app.state.auth
    for username, (_, role) in PASSWORDS.items():
        token = sign_in(app, username).cookies.get("bf_session")
        session = verify_session(token, state.key, now=int(state.clock()), is_revoked=state.denylist.contains)
        assert session is not None
        assert session.principal == Principal(username=username, role=role, user_id=f"id-{username}")
        assert session.expires_at == int(state.clock()) + 12 * 3600
        assert verify_session(token, state.key, now=int(state.clock()) + 12 * 3600 + 31) is None


def test_me_without_a_session_is_401(app) -> None:
    r = client_for(app).get("/api/auth/me")
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["detail"] == "Sign in to continue."


def test_wrong_password_and_unknown_user_give_the_same_answer(app) -> None:
    client = client_for(app)
    a = client.post("/api/auth/login", json={"username": "op", "password": "nope"})
    b = client.post("/api/auth/login", json={"username": "ghost", "password": "nope"})
    assert a.status_code == b.status_code == 401
    assert a.json()["detail"] == b.json()["detail"] == "Unknown user name or wrong password."
    assert "set-cookie" not in a.headers


def test_unreachable_user_directory_is_503(app) -> None:
    app.state.auth.identity.down = True
    r = client_for(app).post("/api/auth/login", json={"username": "op", "password": "pw-op"})
    assert r.status_code == 503


def test_session_ends_after_12_hours(app) -> None:
    client = sign_in(app, "op")
    app.state.auth.clock.t += 12 * 3600 + 31
    assert client.get("/api/auth/me").status_code == 401


def test_sign_out_revokes_the_token(app) -> None:
    client = sign_in(app, "adm")
    token = client.cookies.get("bf_session")
    assert with_token(app, token).get("/api/auth/me").status_code == 200  # the replay path works before sign-out
    r = client.post("/api/auth/logout")
    assert r.status_code == 204
    cleared = [c.lower() for c in r.headers.get_list("set-cookie")]
    assert len(cleared) == 2 and all('bf_session=""' in c or "max-age=0" in c for c in cleared)
    assert {c.split("path=")[1].split(";")[0] for c in cleared} == {"/", "/api"}  # the cookie of SV-0.4 as well
    assert with_token(app, token).get("/api/auth/me").status_code == 401
    assert client_for(app).post("/api/auth/logout").status_code == 204  # without a session as well


def test_sign_out_is_shared_with_other_processes(app) -> None:
    """Revocations reach the shared store (Redis when configured) and are read from it (SV-6.6, #320)."""

    class Shared:
        def __init__(self) -> None:
            self.revoked: dict[str, float] = {}

        def add(self, jti: str, expires_at: float) -> None:
            self.revoked[jti] = expires_at

        def contains(self, jti: str) -> bool:
            return jti in self.revoked

    state = app.state.auth
    shared = Shared()
    state.denylist = DenyList(clock=state.clock, shared=shared)
    client = sign_in(app, "adm")
    token = client.cookies.get("bf_session")
    jti = decode(token, state.key, now=int(state.clock()))["jti"]
    assert client.post("/api/auth/logout").status_code == 204
    assert shared.revoked == {jti: int(state.clock()) + 12 * 3600}

    other = sign_in(app, "op")  # revoked elsewhere (a Streamlit sign-out): refused here
    shared.add(decode(other.cookies.get("bf_session"), state.key, now=int(state.clock()))["jti"], 0)
    assert other.get("/api/auth/me").status_code == 401


def test_token_signed_with_another_key_is_refused(app) -> None:
    other = build()
    other.state.auth.key = b"z" * 32
    token = sign_in(other, "adm").cookies.get("bf_session")
    assert with_token(other, token).get("/api/auth/me").status_code == 200
    assert with_token(app, token).get("/api/auth/me").status_code == 401


# ---------------------------------------------------------------- failure limit


def test_sixth_failed_sign_in_within_a_minute_is_refused() -> None:
    app = build(trusted_proxies="testclient")
    client = client_for(app)
    attacker = {"X-Forwarded-For": "10.0.0.5"}
    for _ in range(5):
        r = client.post("/api/auth/login", json={"username": "op", "password": "guess"}, headers=attacker)
        assert r.status_code == 401
    calls = app.state.auth.identity.calls
    sixth = client.post("/api/auth/login", json={"username": "op", "password": "pw-op"}, headers=attacker)
    assert sixth.status_code == 429
    assert 0 < int(sixth.headers["retry-after"]) <= 60
    assert app.state.auth.identity.calls == calls  # refused before the credentials are checked

    other = client.post(
        "/api/auth/login", json={"username": "op", "password": "pw-op"}, headers={"X-Forwarded-For": "10.0.0.6"}
    )
    assert other.status_code == 200

    app.state.limiter_clock.t += 61  # the oldest failures leave the window
    again = client.post("/api/auth/login", json={"username": "op", "password": "pw-op"}, headers=attacker)
    assert again.status_code == 200


def test_forwarded_for_is_ignored_from_untrusted_peers(app) -> None:
    client = client_for(app)  # peer "testclient" is not a trusted proxy by default
    for i in range(5):
        headers = {"X-Forwarded-For": f"10.0.1.{i}"}
        assert (
            client.post("/api/auth/login", json={"username": "op", "password": "x"}, headers=headers).status_code == 401
        )
    assert client.post("/api/auth/login", json={"username": "op", "password": "pw-op"}).status_code == 429


def test_failure_limiter_window() -> None:
    clock = Clock(0.0)
    limiter = FailureLimiter(2, 60, clock=clock)
    limiter.record_failure("a")
    clock.t = 30
    limiter.record_failure("a")
    assert limiter.retry_after("a") == pytest.approx(30)
    clock.t = 60.5
    assert limiter.retry_after("a") == 0


# ---------------------------------------------------------------- origin guard


def test_cross_site_post_is_refused(app) -> None:
    client = client_for(app)
    body = {"username": "op", "password": "pw-op"}
    r = client.post("/api/auth/login", json=body, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403 and r.json()["detail"].startswith("Requests that change data")
    assert client.post("/api/auth/login", json=body, headers={"Origin": "https://testserver"}).status_code == 200


# ---------------------------------------------------------------- roles


ROLE_MATRIX = {
    # actor:      user  supervisor admin   (endpoint minimum role)
    None: (401, 401, 401),
    "op": (200, 403, 403),
    "sup": (200, 200, 403),
    "adm": (200, 200, 200),
}


@pytest.mark.parametrize("actor", list(ROLE_MATRIX))
def test_role_by_endpoint_matrix(app, actor) -> None:
    client = sign_in(app, actor) if actor else client_for(app)
    got = tuple(client.get(f"/api/_probe/{role}").status_code for role in ("user", "supervisor", "admin"))
    assert got == ROLE_MATRIX[actor]


def test_admin_routes_answer_403_to_user_and_supervisor(app) -> None:
    for actor in ("op", "sup"):
        r = sign_in(app, actor).get("/api/_probe/admin")
        assert r.status_code == 403
        assert r.json()["detail"].startswith("This action needs the admin role")


def _declared_roles(dependant) -> list[str]:
    roles = []
    for dep in dependant.dependencies:
        role = required_role_of(dep.call)
        if role:
            roles.append(role)
        roles.extend(_declared_roles(dep))
    return roles


def test_every_api_route_outside_the_public_list_requires_a_role() -> None:
    app = create_app(ApiSettings(static_dir=None, jwt_secret=SECRET))
    routes = [r for r in app.routes if isinstance(r, APIRoute) and r.path.startswith("/api/")]
    assert {r.path for r in routes} >= PUBLIC_PATHS
    missing = [r.path for r in routes if r.path not in PUBLIC_PATHS and not _declared_roles(r.dependant)]
    assert not missing, f"routes without require_role: {missing}"


# ---------------------------------------------------------------- settings


def test_prod_requires_a_signing_key() -> None:
    with pytest.raises(RuntimeError, match="BF_API_JWT_SECRET"):
        create_app(ApiSettings(static_dir=None, environment="prod", jwt_secret=None))


def test_short_signing_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ApiSettings(jwt_secret="too-short")
