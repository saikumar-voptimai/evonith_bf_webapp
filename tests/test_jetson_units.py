"""The Jetson's service units (SV-6.3, #317): ``deploy/jetson/systemd``, the health check ``healthcheck.sh`` and the
services step of ``provision.sh``.

The units run on the Orin; this file reads them the way systemd does (sections, repeated keys) and checks the keys
every service needs, the ports against the Caddyfile and the environment template, the start order, and the release
pinning of the Python units. The health check's pure functions run in bash after sourcing the script, and one test
runs its main loop against fake ``systemctl``, ``curl`` and ``redis-cli``. Bash: Git for Windows on Windows, else
``BF_BASH_BIN`` (``tests/test_jetson_scripts.py``).
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

import pytest

from tests.test_jetson_scripts import BASH, _bash, _call, _snippet, _status

ROOT = Path(__file__).resolve().parents[1]
JETSON = ROOT / "deploy" / "jetson"
UNITS_DIR = JETSON / "systemd"
CADDYFILE = JETSON / "Caddyfile"
HEALTHCHECK = JETSON / "healthcheck.sh"
PROVISION = JETSON / "provision.sh"
RELEASE = PurePosixPath("/opt/bf-webapp/current")
TARGET = "bf-webapp.target"
# long-running services and the address each listens on (None: no listener)
SERVERS = {
    "bf-redis.service": "127.0.0.1:6379",
    "bf-furnace-data.service": "127.0.0.1:8000",
    "bf-twin-api.service": "127.0.0.1:8100",
    "bf-twin-worker.service": None,
    "bf-streamlit.service": "127.0.0.1:8501",
    "bf-caddy.service": ":443",
}
PYTHON_UNITS = ("bf-furnace-data.service", "bf-twin-api.service", "bf-twin-worker.service", "bf-streamlit.service")
MEMORY_BUDGET_GIB = 24.5  # AGX Orin 32 GB shared with the GPU, less the system (README, "Services")
needs_bash = pytest.mark.requires_service(BASH is not None, "bash is not installed (Git for Windows, or BF_BASH_BIN)")

KNOWN_KEYS = {
    "Unit": {
        "Description",
        "Documentation",
        "After",
        "Wants",
        "Requires",
        "RequiresMountsFor",
        "ConditionPathExists",
        "PartOf",
        "StartLimitIntervalSec",
    },
    "Service": {
        "Type",
        "User",
        "Group",
        "EnvironmentFile",
        "Environment",
        "WorkingDirectory",
        "StateDirectory",
        "StateDirectoryMode",
        "RuntimeDirectory",
        "RuntimeDirectoryPreserve",
        "ExecStart",
        "ExecReload",
        "Restart",
        "RestartSec",
        "KillMode",
        "TimeoutStartSec",
        "TimeoutStopSec",
        "MemoryMax",
        "CPUWeight",
        "LimitNOFILE",
        "LogLevelMax",
        "AmbientCapabilities",
        "CapabilityBoundingSet",
        "NoNewPrivileges",
        "PrivateTmp",
        "ProtectSystem",
        "ProtectHome",
    },
    "Timer": {"OnActiveSec", "OnUnitInactiveSec", "AccuracySec", "Unit"},
    "Install": {"WantedBy"},
}
LIST_KEYS = {"After", "Wants", "Requires", "PartOf", "WantedBy", "RequiresMountsFor"}


# ---- reading units --------------------------------------------------------------------------------------------------


def parse_unit(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Sections of a unit file as ``{section: [(key, value), ...]}``, repeated keys kept in order."""
    sections: dict[str, list[tuple[str, str]]] = {}
    current: str | None = None
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            assert current not in sections, f"{path.name}:{number}: section [{current}] twice"
            sections[current] = []
            continue
        key, sep, value = line.partition("=")
        assert current is not None and sep, f"{path.name}:{number}: not a key=value line in a section: {raw!r}"
        assert not raw.endswith("\\"), f"{path.name}:{number}: no line continuations"
        sections[current].append((key.strip(), value.strip()))
    return sections


UNITS = {path.name: parse_unit(path) for path in sorted(UNITS_DIR.iterdir())}


def values(unit: str, section: str, key: str) -> list[str]:
    """Every value of ``key``; list keys (After=, Wants=, ...) split into their items."""
    found = [v for k, v in UNITS[unit].get(section, []) if k == key]
    return [item for v in found for item in v.split()] if key in LIST_KEYS else found


def value(unit: str, section: str, key: str) -> str:
    found = values(unit, section, key)
    assert len(found) == 1, f"{unit} [{section}] {key}: {found}"
    return found[0]


def environment(unit: str) -> dict[str, str]:
    return dict(item.split("=", 1) for line in values(unit, "Service", "Environment") for item in line.split())


def exec_args(unit: str) -> list[str]:
    return value(unit, "Service", "ExecStart").split()


def option(args: list[str], name: str) -> str:
    assert args.count(name) == 1, f"{name} in {args}"
    return args[args.index(name) + 1]


def provision_array(name: str) -> list[str]:
    match = re.search(rf"^{name}=\(([^)]*)\)", PROVISION.read_text(encoding="utf-8"), re.M)
    assert match, f"{name}=( ... ) not in provision.sh"
    return match.group(1).split()


def provision_value(name: str) -> str:
    match = re.search(rf"^{name}=(\S+)$", PROVISION.read_text(encoding="utf-8"), re.M)
    assert match, f"{name}= not in provision.sh"
    return match.group(1)


def seconds(text: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|min)?", text)
    assert match, text
    return float(match.group(1)) * {"ms": 0.001, "s": 1, "min": 60, None: 1}[match.group(2)]


def gib(text: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMG])", text)
    assert match, text
    return float(match.group(1)) * {"K": 2**-20, "M": 2**-10, "G": 1}[match.group(2)]


def caddy_default(variable: str) -> str:
    match = re.search(rf"\{{\${variable}:([^}}]+)\}}", CADDYFILE.read_text(encoding="utf-8"))
    assert match, f"{{${variable}:...}} not in the Caddyfile"
    return match.group(1)


# ---- the files ------------------------------------------------------------------------------------------------------


def test_provision_installs_exactly_the_unit_files() -> None:
    assert sorted(provision_array("UNITS")) == sorted(UNITS)
    assert set(SERVERS) | {TARGET, "bf-healthcheck.service", "bf-healthcheck.timer"} == set(UNITS)
    assert provision_value("SERVICES_TARGET") == TARGET
    assert provision_array("PACKAGE_UNITS") == ["redis-server.service", "caddy.service"]  # masked: 6379, 443
    assert "redis-server" in provision_array("PACKAGES") and "curl" in provision_array("PACKAGES")


def test_unit_files_are_lf_with_known_sections_and_keys() -> None:
    for name, sections in UNITS.items():
        raw = (UNITS_DIR / name).read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n"), f"{name}: LF line ends (.gitattributes)"
        kind = name.rsplit(".", 1)[1]
        allowed = {"service": {"Unit", "Service", "Install"}, "timer": {"Unit", "Timer", "Install"}}.get(
            kind, {"Unit", "Install"}
        )
        assert "Unit" in sections and set(sections) <= allowed, (name, list(sections))
        for section, pairs in sections.items():
            unknown = sorted({k for k, _ in pairs} - KNOWN_KEYS[section])
            assert not unknown, f"{name} [{section}]: unknown keys {unknown} (typo, or add them to KNOWN_KEYS)"
        assert value(name, "Unit", "Description").startswith("BF-02 webapp: ")
    git = shutil.which("git")
    if git:
        attrs = subprocess.run(
            [git, "check-attr", "eol", "--", *(f"systemd/{n}" for n in UNITS)],
            cwd=JETSON,
            capture_output=True,
            text=True,
            check=False,
        )
        if attrs.returncode == 0:
            assert attrs.stdout.count(": eol: lf") == len(UNITS), attrs.stdout


def test_every_service_restarts_runs_as_bfapp_and_has_a_memory_limit() -> None:
    total = 0.0
    for unit in SERVERS:
        assert value(unit, "Service", "User") == "bfapp" and value(unit, "Service", "Group") == "bfapp", unit
        assert values(unit, "Service", "EnvironmentFile")[0] == "/etc/bf-webapp/env", unit  # required: no "-"
        assert value(unit, "Service", "Restart") == "always", unit
        assert seconds(value(unit, "Service", "RestartSec")) <= 5, unit
        assert value(unit, "Unit", "StartLimitIntervalSec") == "0", unit  # never gives up restarting
        assert seconds(value(unit, "Service", "TimeoutStopSec")) <= 30, unit
        assert values(unit, "Unit", "RequiresMountsFor") == ["/var/lib/bf-webapp"], unit  # never on the eMMC
        assert values(unit, "Unit", "PartOf") == [TARGET] and values(unit, "Install", "WantedBy") == [TARGET], unit
        for key, want in (("NoNewPrivileges", "yes"), ("PrivateTmp", "yes"), ("ProtectSystem", "full")):
            assert value(unit, "Service", key) == want, (unit, key)
        total += gib(value(unit, "Service", "MemoryMax"))
    assert total <= MEMORY_BUDGET_GIB, f"MemoryMax adds up to {total} GiB"
    assert gib(value("bf-twin-worker.service", "Service", "MemoryMax")) >= 8  # pool of 8 + STRATA-Flow solvers
    assert values(TARGET, "Install", "WantedBy") == ["multi-user.target"]


def test_ports_agree_with_the_caddyfile_and_the_environment() -> None:
    api = exec_args("bf-twin-api.service")
    assert f"{option(api, '--host')}:{option(api, '--port')}" == caddy_default("BF_API_UPSTREAM") == "127.0.0.1:8100"
    assert option(api, "--workers") == "2" and option(api, "--app-dir") == "src"
    assert "api.main:app" in api and int(option(api, "--timeout-graceful-shutdown")) < seconds(
        value("bf-twin-api.service", "Service", "TimeoutStopSec")
    )  # open event streams end before systemd kills

    legacy = exec_args("bf-streamlit.service")
    address = f"{option(legacy, '--server.address')}:{option(legacy, '--server.port')}"
    assert address == caddy_default("BF_LEGACY_UPSTREAM") == "127.0.0.1:8501"
    assert option(legacy, "--server.baseUrlPath") == "legacy"
    assert re.search(r"^\s*@legacy path /legacy /legacy/\*$", CADDYFILE.read_text(encoding="utf-8"), re.M)
    assert option(legacy, "--server.headless") == "true" and option(legacy, "--server.fileWatcherType") == "none"

    caddy = exec_args("bf-caddy.service")
    config = PurePosixPath(option(caddy, "--config"))
    assert config == RELEASE / "deploy/jetson/Caddyfile" and option(caddy, "--adapter") == "caddyfile"
    assert values("bf-caddy.service", "Unit", "ConditionPathExists") == [str(config)]
    assert caddy_default("BF_HTTPS_PORT") == "443" and caddy_default("BF_HTTP_PORT") == "80"
    assert value("bf-caddy.service", "Service", "AmbientCapabilities") == "CAP_NET_BIND_SERVICE"  # 80, 443 as bfapp
    assert caddy_default("BF_SPA_ROOT") == f"{RELEASE}/frontend/build"

    redis = exec_args("bf-redis.service")
    assert f"{option(redis, '--bind')}:{option(redis, '--port')}" == SERVERS["bf-redis.service"]
    template = _call(PROVISION, "render_env_template", "ab" * 32) if BASH else PROVISION.read_text(encoding="utf-8")
    url = re.search(r"^BF_REDIS_URL=redis://([^/\s]+)/\d+$", template, re.M)
    assert url and url.group(1) == SERVERS["bf-redis.service"]
    assert option(redis, "--supervised") == "systemd" and value("bf-redis.service", "Service", "Type") == "notify"
    assert option(redis, "--appendonly") == "yes" and option(redis, "--maxmemory-policy") == "noeviction"
    assert PurePosixPath(option(redis, "--dir")) == PurePosixPath("/var/lib") / value(
        "bf-redis.service", "Service", "StateDirectory"
    )

    data = exec_args("bf-furnace-data.service")
    assert f"{option(data, '--host')}:{option(data, '--port')}" == SERVERS["bf-furnace-data.service"]

    listening = [a for a in SERVERS.values() if a]
    assert len({a.rsplit(":", 1)[1] for a in listening}) == len(listening)  # one port per service
    exposed = [a for a in listening if not a.startswith("127.0.0.1:")]
    assert exposed == [":443"]  # only Caddy beyond loopback (ufw, verify_base.sh)


def test_python_units_run_the_release_resolved_at_their_start() -> None:
    for unit in PYTHON_UNITS:
        cwd = PurePosixPath(value(unit, "Service", "WorkingDirectory"))
        assert cwd == RELEASE or cwd.parent == RELEASE, unit
        args = exec_args(unit)
        assert args[0] == "/usr/bin/env" and not args[1].startswith("/"), unit  # relative: pinned at the start
        resolved = PurePosixPath(re.sub(r"/[^/]+/\.\./", "/", str(cwd / args[1])))
        assert resolved == RELEASE / ".venv/bin/python", unit
        assert values(unit, "Unit", "ConditionPathExists") == [str(resolved)], unit  # no release: not started
        assert "/etc/bf-webapp/env" in values(unit, "Service", "EnvironmentFile")
        assert "-/etc/bf-webapp/uv.env" in values(unit, "Service", "EnvironmentFile")
    worker = exec_args("bf-twin-worker.service")
    assert worker[2:] == ["-m", "worker"] and (ROOT / "src" / "worker" / "__main__.py").is_file()
    env = environment("bf-twin-worker.service")
    assert env["PYTHONPATH"] == "src" and env["BF_WORKER_POOL"] == "8"
    assert value("bf-twin-worker.service", "Service", "KillMode") == "mixed"
    assert (ROOT / "src" / "api" / "main.py").is_file() and (ROOT / "run_streamlit.py").is_file()
    assert exec_args("bf-streamlit.service")[2] == "run_streamlit.py"
    assert (ROOT / "furnace-data-service" / "app" / "main.py").is_file()


def test_start_order_brings_the_upstreams_first_without_a_cycle() -> None:
    after = {u: {d for d in values(u, "Unit", "After") if d in UNITS} for u in SERVERS}
    assert after["bf-caddy.service"] == {"bf-twin-api.service", "bf-streamlit.service"}
    assert after["bf-twin-api.service"] == {"bf-redis.service"}
    assert after["bf-twin-worker.service"] == {"bf-redis.service"}
    assert after["bf-streamlit.service"] == {"bf-redis.service", "bf-furnace-data.service"}
    assert after["bf-redis.service"] == set() and after["bf-furnace-data.service"] == set()
    for unit, deps in after.items():
        assert deps <= set(values(unit, "Unit", "Wants")), f"{unit}: ordered after units it does not pull in"
        # Requires= between the services: a start that fails on a dependency is never retried (README)
        assert not [d for d in values(unit, "Unit", "Requires") if d in UNITS], unit
    ordered: list[str] = []
    pending = dict(after)
    while pending:
        ready = sorted(u for u, deps in pending.items() if deps <= set(ordered))
        assert ready, f"ordering cycle among {sorted(pending)}"
        ordered += ready
        for u in ready:
            del pending[u]
    assert ordered[-1] == "bf-caddy.service"
    for unit in UNITS:
        for key in ("After", "Wants", "PartOf", "WantedBy"):
            section = "Install" if key == "WantedBy" else "Unit"
            for dep in values(unit, section, key):
                assert not dep.startswith("bf-") or dep in UNITS, f"{unit} {key}={dep}: no such unit"


def test_health_check_timer_probes_every_server() -> None:
    timer = "bf-healthcheck.timer"
    assert value(timer, "Timer", "Unit") == "bf-healthcheck.service"
    assert seconds(value(timer, "Timer", "OnUnitInactiveSec")) <= 10  # after each run ends: runs never overlap
    assert values(timer, "Install", "WantedBy") == [TARGET] and values(timer, "Unit", "PartOf") == [TARGET]
    service = "bf-healthcheck.service"
    assert value(service, "Service", "Type") == "oneshot" and "Install" not in UNITS[service]  # the timer starts it
    assert (
        value(service, "Service", "ExecStart") == provision_value("HEALTHCHECK_BIN") == "/usr/local/sbin/bf-healthcheck"
    )
    assert value(service, "Service", "RuntimeDirectory") == "bf-healthcheck"
    assert "STATE_DIR=/run/bf-healthcheck " in HEALTHCHECK.read_text(encoding="utf-8")
    assert value(service, "Service", "LogLevelMax") == "warning"


# ---- the health check -----------------------------------------------------------------------------------------------


def _probes() -> dict[str, list[str]]:
    lines = _snippet(HEALTHCHECK, 'printf "%s\\n" "${PROBES[@]}"').stdout.splitlines()
    return {line.split()[0]: line.split()[1:] for line in lines}


@needs_bash
def test_health_check_is_valid_bash_and_executable() -> None:
    raw = HEALTHCHECK.read_bytes()
    assert b"\r" not in raw and raw.startswith(b"#!/usr/bin/env bash\n")
    done = _bash("-n", HEALTHCHECK.as_posix())
    assert done.returncode == 0, done.stderr
    git = shutil.which("git")
    if git:
        listed = subprocess.run([git, "ls-files", "-s", HEALTHCHECK.name], cwd=JETSON, capture_output=True, text=True)
        if listed.returncode == 0 and listed.stdout:
            assert listed.stdout.startswith("100755 "), "not executable in git (git add --chmod=+x)"


@needs_bash
def test_health_check_probes_each_server_on_its_port() -> None:
    probes = _probes()
    assert set(probes) == set(SERVERS)
    api_health = probes["bf-twin-api.service"][2]
    assert api_health == f"http://{SERVERS['bf-twin-api.service']}/api/health"
    from api.auth import PUBLIC_PATHS

    assert "/api/health" in PUBLIC_PATHS  # answers without a session
    assert probes["bf-streamlit.service"][2] == f"http://{SERVERS['bf-streamlit.service']}/legacy/_stcore/health"
    assert probes["bf-furnace-data.service"][2] == f"http://{SERVERS['bf-furnace-data.service']}/health"
    health_route = (ROOT / "furnace-data-service" / "app" / "routes" / "health.py").read_text(encoding="utf-8")
    assert '@router.get("/health"' in health_route
    assert [probes[u][1] for u in ("bf-redis.service", "bf-caddy.service", "bf-twin-worker.service")] == [
        "redis",
        "https",
        "heartbeat",
    ]
    for unit, (grace, *_rest) in probes.items():
        assert 10 <= int(grace) <= 120, unit
    assert int(probes["bf-twin-api.service"][0]) >= 60  # imports take ~16 s on the laptop
    text = HEALTHCHECK.read_text(encoding="utf-8")
    from worker.runtime import ALIVE_KEY, HEARTBEAT_S

    assert f"WORKER_HEARTBEAT_KEY={ALIVE_KEY}\n" in text
    max_age = int(re.search(r"^WORKER_HEARTBEAT_MAX_S=(\d+)$", text, re.M).group(1))
    assert max_age >= 5 * HEARTBEAT_S


@needs_bash
def test_health_check_restarts_after_two_failed_probes_in_a_row() -> None:
    def decide(*args: object) -> str:
        return _call(HEALTHCHECK, "decide", *map(str, args)).strip()

    # state age grace probe fails limit
    assert decide("active", 100, 90, "ok", 1, 2) == "ok 0"
    assert decide("active", 100, 90, "fail", 0, 2) == "fail 1"
    assert decide("active", 100, 90, "fail", 1, 2) == "restart 0"
    assert decide("active", 30, 90, "fail", 1, 2) == "idle 0"  # still starting
    assert decide("activating", 300, 90, "fail", 1, 2) == "idle 0"
    assert decide("inactive", 300, 90, "fail", 1, 2) == "idle 0"  # stopped by hand
    assert decide("failed", 300, 90, "fail", 1, 2) == "idle 0"  # systemd restarts it
    assert decide("active", 300, 120, "skip", 1, 2) == "idle 0"  # worker while Redis does not answer
    assert decide("active", 300, 90, "fail", 0, 1) == "restart 0"


@needs_bash
def test_health_check_ages() -> None:
    assert _call(HEALTHCHECK, "unit_age_s", "5000000", "125.5").strip() == "120"
    assert _call(HEALTHCHECK, "unit_age_s", "0", "125.5").strip() == "0"  # never active
    assert _call(HEALTHCHECK, "unit_age_s", "200000000", "125.5").strip() == "0"
    now = datetime(2026, 10, 29, 4, 30, 15, 123456, tzinfo=UTC)
    beat = (now - timedelta(seconds=42)).isoformat()  # the worker's mark_alive format
    assert _call(HEALTHCHECK, "heartbeat_age_s", beat, str(int(now.timestamp()))).strip() == "42"
    assert _status(HEALTHCHECK, "heartbeat_age_s", "not a time", "0") == 1


FAKE_SYSTEMCTL = """#!/usr/bin/env bash
case $1 in
is-active) cat "$FAKE/state.$2" 2>/dev/null || echo inactive ;;
show) echo 1000000 ;;
restart) echo "$3" >>"$FAKE/restarts" ;;
esac
"""
FAKE_CURL = """#!/usr/bin/env bash
for a in "$@"; do
	case $a in
	*:8100/*) exit 7 ;;
	https://*) printf 502; exit 0 ;;
	esac
done
exit 0
"""
FAKE_REDIS = """#!/usr/bin/env bash
case "$*" in
*ping*) echo PONG ;;
*get*) date -u -d '-5 seconds' +%Y-%m-%dT%H:%M:%S.000000+00:00 ;;
esac
"""


@needs_bash
def test_health_check_main_restarts_only_the_service_that_stopped_answering(tmp_path: Path) -> None:
    fake = tmp_path / "fake"
    fake.mkdir()
    for name, text in (("systemctl", FAKE_SYSTEMCTL), ("curl", FAKE_CURL), ("redis-cli", FAKE_REDIS)):
        (fake / name).write_bytes(text.encode())
        (fake / name).chmod(0o755)
    for unit in SERVERS:
        (fake / f"state.{unit}").write_bytes(b"active\n")
    (fake / "uptime").write_bytes(b"3600.00 100.00\n")  # active for 3599 s: past every grace
    code = (
        'd=$1; if command -v cygpath >/dev/null; then d=$(cygpath -u "$d"); fi; export FAKE=$d PATH="$d:$PATH"; '
        'STATE_DIR=$d/state UPTIME_FILE=$d/uptime; main; echo "--"; main; echo "--"; main'
    )
    done = _snippet(HEALTHCHECK, code, str(fake))
    assert done.returncode == 0, done.stderr
    first, second, third = done.stdout.split("--\n")
    assert first == "<4>bf-twin-api.service: http probe failed (1 of 2 in a row)\n"
    assert second == "<4>bf-twin-api.service: 2 http probes failed in a row, restarting it\n"
    assert third == first  # the count starts again after a restart
    assert (fake / "restarts").read_text().split() == ["bf-twin-api.service"]
    assert (fake / "state" / "bf-caddy.service").read_text().strip() == "0"  # a 502 is an answer: Caddy serves


# ---- provision.sh: the services step --------------------------------------------------------------------------------


@needs_bash
def test_caddy_release_asset_version_and_checksum(tmp_path: Path) -> None:
    assert _call(PROVISION, "caddy_asset", "aarch64", "2.10.2").strip() == "caddy_2.10.2_linux_arm64.tar.gz"
    assert _status(PROVISION, "caddy_asset", "riscv64", "2.10.2") == 1
    assert _call(PROVISION, "caddy_version", stdin="v2.10.2 h1:Iz6Df3QJ5V0D7a4w=\n").strip() == "2.10.2"
    assert _status(PROVISION, "caddy_version", stdin="caddy: command not found\n") == 1
    tarball = tmp_path / "caddy_2.10.2_linux_arm64.tar.gz"
    tarball.write_bytes(b"caddy binary")
    import hashlib

    sha512 = hashlib.sha512(b"caddy binary").hexdigest()
    sha256 = hashlib.sha256(b"caddy binary").hexdigest()
    other = hashlib.sha512(b"other").hexdigest()
    listing = f"{other}  caddy_2.10.2_linux_amd64.tar.gz\n{sha512}  {tarball.name}\n"
    assert _status(PROVISION, "checksum_listed", tarball.as_posix(), tarball.name, stdin=listing) == 0
    assert (
        _status(PROVISION, "checksum_listed", tarball.as_posix(), tarball.name, stdin=f"{sha256}  {tarball.name}\n")
        == 0
    )
    assert (
        _status(PROVISION, "checksum_listed", tarball.as_posix(), tarball.name, stdin=f"{other}  {tarball.name}\n") == 1
    )
    assert _status(PROVISION, "checksum_listed", tarball.as_posix(), tarball.name, stdin=f"{other}  x.tar.gz\n") == 1
    assert provision_value("CADDY_VERSION") == "2.10.2"  # the version the Caddyfile was verified with (README)


@needs_bash
def test_units_with_an_install_section_are_enabled() -> None:
    enabled = [u for u in UNITS if _status(PROVISION, "has_install", stdin=(UNITS_DIR / u).read_text()) == 0]
    assert sorted(enabled) == sorted(set(UNITS) - {"bf-healthcheck.service"})


# ---- run_streamlit.py -----------------------------------------------------------------------------------------------


def test_run_streamlit_imports_torch_only_on_windows_and_runs_app_py_from_the_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location("run_streamlit_under_test", ROOT / "run_streamlit.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.APP == ROOT / "src" / "app.py" and module.APP.is_file()  # from the file, not the working directory
    calls: list[tuple[list[str], str]] = []
    cli = types.ModuleType("streamlit.web.cli")
    cli.main = lambda: calls.append((list(sys.argv), os.getcwd())) or 0  # type: ignore[attr-defined]
    web = types.ModuleType("streamlit.web")
    web.cli = cli  # type: ignore[attr-defined]
    package = types.ModuleType("streamlit")
    package.web = web  # type: ignore[attr-defined]
    for name, stub in (("streamlit", package), ("streamlit.web", web), ("streamlit.web.cli", cli)):
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setitem(sys.modules, "torch", None)  # importing torch would raise ImportError
    monkeypatch.setattr(module.os, "name", "posix")  # Path() would now build a PosixPath: strings below
    monkeypatch.chdir(ROOT / "deploy")
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    assert module.main(["--server.port", "8501", "--server.baseUrlPath", "legacy"]) == 0
    argv = ["streamlit", "run", str(ROOT / "src" / "app.py"), "--server.port", "8501", "--server.baseUrlPath", "legacy"]
    assert calls == [(argv, str(ROOT))]  # run from the repository root
    source = (ROOT / "run_streamlit.py").read_text(encoding="utf-8")
    assert re.search(r'if os\.name == "nt":\n\s+import torch', source)
