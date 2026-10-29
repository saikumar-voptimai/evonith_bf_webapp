"""The Jetson scripts: base system (SV-6.1, #315: ``provision.sh``, ``verify_base.sh``), Python and benchmark (SV-6.2,
#316: ``install_python.sh``, ``benchmark.sh``).

The scripts run on the Orin (the first three as root), so this file never runs a system step; ``benchmark.sh`` runs
end to end on a fake release with a fake uv. It checks the syntax (``bash -n``), the steps and
their order, the argument checks (they stop before the first step), and the pure functions the scripts define: the
rendered configuration files, the JetPack / nvpmodel / fstab / CIDR helpers and the parsers of ``chronyc tracking``,
``ufw status`` and ``ss``. Each function is called in bash after sourcing the script (``main`` runs only when the file
is executed). Needs bash: Git for Windows on Windows (not the WSL launcher), else ``BF_BASH_BIN``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JETSON = ROOT / "deploy" / "jetson"
PROVISION = JETSON / "provision.sh"
VERIFY = JETSON / "verify_base.sh"
INSTALL = JETSON / "install_python.sh"
BENCH = JETSON / "benchmark.sh"
SCRIPTS = [PROVISION, VERIFY, INSTALL, BENCH]
UUID = "0f3a1b2c-1111-2222-3333-444455556666"


def _find_bash() -> str | None:
    if os.environ.get("BF_BASH_BIN"):
        return os.environ["BF_BASH_BIN"]
    if os.name != "nt":
        return shutil.which("bash")
    # C:\Windows\System32\bash.exe starts WSL; Git for Windows' bin\bash.exe sets up its own PATH (awk, sed, stat).
    candidates: list[Path] = []
    git = shutil.which("git")
    if git:
        root = Path(git).resolve().parent.parent  # Git\cmd\git.exe, Git\bin\git.exe or Git\mingw64\bin\git.exe
        candidates += [root / "bin" / "bash.exe", root.parent / "bin" / "bash.exe"]
    candidates += [
        Path(os.environ[v]) / "Git" / "bin" / "bash.exe" for v in ("ProgramFiles", "ProgramW6432") if v in os.environ
    ]
    return next((str(c) for c in candidates if c.is_file()), None)


BASH = _find_bash()
pytestmark = pytest.mark.requires_service(
    BASH is not None, "bash is not installed (Git for Windows, or set BF_BASH_BIN)"
)


def _bash(*args: str, stdin: str = "") -> subprocess.CompletedProcess[str]:
    """Bytes in and out: text mode would turn ``\\n`` into ``\\r\\n`` on the way to bash on Windows."""
    assert BASH is not None
    done = subprocess.run([BASH, *args], input=stdin.encode(), capture_output=True, timeout=60, check=False)
    return subprocess.CompletedProcess(done.args, done.returncode, done.stdout.decode(), done.stderr.decode())


def _snippet(script: Path, code: str, *args: str, stdin: str = "") -> subprocess.CompletedProcess[str]:
    """Run ``code`` in bash after sourcing ``script``; ``$1..`` are ``args``."""
    return _bash("-c", f'source "$1"; shift; {code}', "bash", script.as_posix(), *args, stdin=stdin)


def _call(script: Path, *command: str, stdin: str = "") -> str:
    done = _snippet(script, '"$@"', *command, stdin=stdin)
    assert done.returncode == 0, f"{command}: rc {done.returncode}\n{done.stderr}"
    return done.stdout


def _status(script: Path, *command: str, stdin: str = "") -> int:
    return _snippet(script, '"$@"', *command, stdin=stdin).returncode


def _function_body(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", text, re.S | re.M)
    assert match, f"{name}() not defined"
    return match.group(1)


# ---- both scripts ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_is_valid_bash_with_lf_line_ends(script: Path) -> None:
    raw = script.read_bytes()
    assert b"\r" not in raw, "CRLF line ends: bash on the Jetson fails on them"
    assert raw.startswith(b"#!/usr/bin/env bash\n")
    done = _bash("-n", script.as_posix())
    assert done.returncode == 0, done.stderr
    git = shutil.which("git")
    if git:
        listed = subprocess.run([git, "ls-files", "-s", script.name], cwd=JETSON, capture_output=True, text=True)
        if listed.returncode == 0 and listed.stdout:
            assert listed.stdout.startswith("100755 "), "not executable in git (git add --chmod=+x)"


def test_both_scripts_hold_the_same_layout_table() -> None:
    code = 'printf "%s\\n" "${LAYOUT_DIRS[@]}"'
    provision = _snippet(PROVISION, code).stdout.splitlines()
    verify = _snippet(VERIFY, code).stdout.splitlines()
    assert provision == verify
    assert provision == [
        "/opt/bf-webapp root:root 755",
        "/opt/bf-webapp/releases root:root 755",
        "/var/lib/bf-webapp bfapp:bfapp 750",
        "/var/lib/bf-webapp/storage bfapp:bfapp 750",
        "/var/lib/bf-webapp/cache bfapp:bfapp 750",
        "/var/log/bf-webapp bfapp:bfapp 750",
        "/etc/bf-webapp root:bfapp 750",
    ]
    for script in (PROVISION, VERIFY):
        assert _snippet(script, 'printf "%s %s" "$ENV_FILE" "$ENV_FILE_MODE"').stdout == "/etc/bf-webapp/env 640"


def test_readme_documents_every_option() -> None:
    readme = (JETSON / "README.md").read_text(encoding="utf-8")
    for script in SCRIPTS:
        usage = _call(script, "usage")
        options = set(re.findall(r"--[a-z][a-z-]+", usage)) - {"--help"}
        assert options, script.name
        missing = sorted(o for o in options if o not in readme)
        assert not missing, f"{script.name} options missing from README: {missing}"
    assert all(script.name in readme for script in SCRIPTS) and "nmap" in readme


# ---- provision.sh: steps and arguments ------------------------------------------------------------------------------


def test_provision_runs_every_required_step_in_order() -> None:
    text = PROVISION.read_text(encoding="utf-8")
    steps = re.findall(r"^\t(step_\w+)$", _function_body(text, "main"), re.M)
    assert steps == [
        "step_platform",  # JetPack 6.x
        "step_packages",
        "step_user",  # bfapp before the mount and the layout that it owns
        "step_nvme",  # mounted before the layout creates its directories on it
        "step_layout",
        "step_power",
        "step_clock",
        "step_logs",
        "step_services",  # units (SV-6.3) after the user, the NVMe and the env file they use
        "step_firewall",  # last: a firewall mistake must not stop the other steps
    ]
    body = {name: _function_body(text, name) for name in steps}
    assert "l4t_version" in body["step_platform"] and "jetpack6" in body["step_platform"]
    assert "useradd --system" in body["step_user"] and "/usr/sbin/nologin" in body["step_user"]
    assert (
        "blkid -U" in body["step_nvme"]
        and "/etc/fstab" in body["step_nvme"]
        and 'mount "$DATA_DIR"' in body["step_nvme"]
    )
    assert "LAYOUT_DIRS" in body["step_layout"] and "$ENV_FILE_MODE" in body["step_layout"]
    assert "root:$APP_USER" in body["step_layout"] and "[[ ! -e $ENV_FILE ]]" in body["step_layout"]  # created once
    assert "maxn_mode" in body["step_power"] and 'systemctl enable "$POWER_UNIT"' in body["step_power"]
    assert "render_chrony_conf" in body["step_clock"] and "chronyc waitsync" in body["step_clock"]
    assert "render_logrotate" in body["step_logs"] and "logrotate --debug" in body["step_logs"]
    assert "install_caddy" in body["step_services"] and 'systemctl mask "$unit"' in body["step_services"]
    assert (
        'write_file "$UNIT_DIR/$unit"' in body["step_services"] and "systemctl daemon-reload" in body["step_services"]
    )
    assert (
        'systemctl enable --quiet "$unit"' in body["step_services"]
        and "[[ ! -L $OPT_DIR/current ]]" in body["step_services"]
    )  # started only with a release
    assert "ufw --force enable" in body["step_firewall"] and "peers_outside" in body["step_firewall"]


def test_help_and_argument_errors_stop_before_the_first_step() -> None:
    assert _bash(PROVISION.as_posix(), "--help").returncode == 0
    valid = ["--nvme-uuid", UUID, "--ntp-server", "ntp.plant.local", "--admin-subnet", "10.20.30.0/24"]
    cases = {
        "--nvme-uuid is required": valid[2:],
        "--ntp-server is required": valid[:2] + valid[4:],
        "--admin-subnet is required": valid[:4],
        "not a filesystem UUID": ["--nvme-uuid", "/dev/nvme0n1p1", *valid[2:]],
        "not a host name": [*valid[:2], "--ntp-server", "ntp;reboot", *valid[4:]],
        "IPv4 network in CIDR form": [*valid[:4], "--admin-subnet", "0.0.0.0/0"],
        "unknown argument": [*valid, "--yes"],
        "needs a value": [*valid, "--ntp-server"],
    }
    for message, args in cases.items():
        done = _bash(PROVISION.as_posix(), *args)
        assert done.returncode == 2, (message, done.stdout, done.stderr)
        assert message in done.stderr and "Usage:" in done.stderr
        assert "==" not in done.stdout  # no step started


def test_arguments_are_normalised() -> None:
    code = (
        'parse_args "$@"; printf "%s|%s|%s|%s|%s" '
        '"$NVME_UUID" "${NTP_SERVERS[*]}" "${ADMIN_SUBNETS[*]}" "$HTTP_REDIRECT" "$ALLOW_SSH_LOCKOUT"'
    )
    args = ["--nvme-uuid", UUID.upper(), "--ntp-server", "ntp1.plant", "--ntp-server", "10.0.0.5"]
    args += ["--admin-subnet", "10.20.30.7/24", "--admin-subnet", "192.168.010.0/24", "--no-http"]
    done = _snippet(PROVISION, code, *args)
    assert done.returncode == 0, done.stderr
    assert done.stdout == f"{UUID}|ntp1.plant 10.0.0.5|10.20.30.0/24 192.168.10.0/24|0|0"


def test_a_user_other_than_root_is_refused_before_the_first_step() -> None:
    if _bash("-c", "echo $EUID").stdout.strip() == "0":
        pytest.skip("running as root: the script would provision this machine")
    valid = ["--nvme-uuid", UUID, "--ntp-server", "ntp.plant.local", "--admin-subnet", "10.20.30.0/24"]
    done = _bash(PROVISION.as_posix(), *valid)
    assert done.returncode == 1 and "run as root" in done.stderr and "==" not in done.stdout
    done = _bash(VERIFY.as_posix())
    assert done.returncode == 1 and "run as root" in done.stderr


# ---- provision.sh: pure helpers -------------------------------------------------------------------------------------


def test_jetpack_6_is_l4t_r36() -> None:
    line = "# R36 (release), REVISION: 4.3, GCID: 38968081, BOARD: generic, EABI: aarch64, DATE: Wed Jan  8 2025\n"
    assert _call(PROVISION, "l4t_version", stdin=line) == "36.4.3\n"
    assert _status(PROVISION, "jetpack6", "36.4.3") == 0
    assert _status(PROVISION, "jetpack6", "35.6.0") == 1  # JetPack 5.1
    assert _status(PROVISION, "l4t_version", stdin="Ubuntu 22.04\n") == 1


def test_maxn_mode_is_read_from_nvpmodel_conf() -> None:
    agx = (
        "< PM_CONFIG DEFAULT=2 >\n< POWER_MODEL ID=0 NAME=MAXN >\nCPU_ONLINE CORE_0 1\n"
        "< POWER_MODEL ID=1 NAME=MODE_15W >\n"
    )
    nano = "< POWER_MODEL ID=0 NAME=15W >\n< POWER_MODEL ID=1 NAME=25W >\n< POWER_MODEL ID=2 NAME=MAXN_SUPER >\n"
    assert _call(PROVISION, "maxn_mode", stdin=agx) == "0 MAXN\n"
    assert _call(PROVISION, "maxn_mode", stdin=nano) == "2 MAXN_SUPER\n"
    assert _status(PROVISION, "maxn_mode", stdin="< POWER_MODEL ID=0 NAME=15W >\n") == 1


def test_power_unit_sets_maxn_then_jetson_clocks_at_boot() -> None:
    unit = _call(PROVISION, "render_power_unit", "/usr/sbin/nvpmodel", "0", "/usr/bin/jetson_clocks")
    assert "Type=oneshot\nRemainAfterExit=yes\n" in unit
    assert unit.index("ExecStart=-/usr/sbin/nvpmodel -m 0") < unit.index("ExecStart=/usr/bin/jetson_clocks")
    assert "After=nvpmodel.service" in unit and "WantedBy=multi-user.target" in unit


def test_fstab_gets_one_entry_by_uuid_and_keeps_the_rest() -> None:
    entry = _call(PROVISION, "render_fstab_line", UUID, "ext4").strip()
    assert entry == f"UUID={UUID} /var/lib/bf-webapp ext4 defaults,noatime,nofail,x-systemd.device-timeout=30s 0 2"
    fstab = "# /etc/fstab: static file system information.\n/dev/root / ext4 defaults 0 1\n"
    once = _call(PROVISION, "fstab_with_entry", entry, stdin=fstab)
    assert once.startswith(fstab) and once.endswith(f"{entry}\n") and once.count("/var/lib/bf-webapp") == 1
    assert _call(PROVISION, "fstab_with_entry", entry, stdin=once) == once  # idempotent
    stale = fstab + "/dev/nvme0n1p1 /var/lib/bf-webapp ext4 defaults 0 2\n#UUID=old /var/lib/bf-webapp ext4 x 0 2\n"
    replaced = _call(PROVISION, "fstab_with_entry", entry, stdin=stale)
    assert "/dev/nvme0n1p1" not in replaced and "#UUID=old" in replaced and replaced.endswith(f"{entry}\n")


def test_chrony_uses_only_the_plant_servers() -> None:
    conf = _call(PROVISION, "render_chrony_conf", "ntp.plant.local", "10.0.0.5")
    servers = [line for line in conf.splitlines() if line.startswith(("server", "pool", "sourcedir", "peer"))]
    assert servers == ["server ntp.plant.local iburst", "server 10.0.0.5 iburst"]
    assert "makestep 1 3" in conf and "rtcsync" in conf


def test_logs_are_kept_14_days() -> None:
    rotate = _call(PROVISION, "render_logrotate")
    assert "/var/log/bf-webapp/*.log {" in rotate
    for line in ("su bfapp bfapp", "daily", "rotate 14", "maxage 14", "missingok", "compress", "copytruncate"):
        assert f"\t{line}\n" in rotate
    journald = _call(PROVISION, "render_journald")
    assert "[Journal]\nStorage=persistent\nMaxRetentionSec=14day\n" in journald


def test_env_template_sets_production_and_leaves_no_empty_value() -> None:
    secret = "ab" * 32
    env = _call(PROVISION, "render_env_template", secret)
    values = dict(line.split("=", 1) for line in env.splitlines() if line and not line.startswith("#"))
    assert values["BF_API_JWT_SECRET"] == secret and len(secret) >= 32
    assert values["BF_API_ENVIRONMENT"] == "prod" and values["BF_API_COOKIE_SECURE"] == "true"
    assert values["BF_REDIS_URL"] == "redis://127.0.0.1:6379/0"
    assert all(values.values()) and not any(k.startswith("export ") for k in values)
    assert "# DATABASE_URL=" in env  # filled in by hand before the first release


def test_cidr_helpers() -> None:
    assert _call(PROVISION, "normalize_cidr", "10.20.30.7/24") == "10.20.30.0/24\n"
    assert _call(PROVISION, "normalize_cidr", "010.1.2.3/32") == "10.1.2.3/32\n"  # decimal, not octal
    for bad in ("0.0.0.0/0", "10.0.0.0/7", "10.0.0.256/24", "10.0.0/24", "fd00::/64", "10.0.0.0"):
        assert _status(PROVISION, "normalize_cidr", bad) == 1, bad
    assert _status(PROVISION, "ip_in_cidr", "10.20.30.99", "10.20.30.0/24") == 0
    assert _status(PROVISION, "ip_in_cidr", "10.20.31.1", "10.20.30.0/24") == 1
    assert _status(PROVISION, "ip_in_cidr", "10.20.31.1", "10.20.0.0/16") == 0


def test_ssh_clients_outside_the_admin_subnets_are_found() -> None:
    ss = (
        "0 0 10.20.30.10:22 10.20.30.15:53122\n"
        "0 0 [::ffff:10.20.30.10]:22 [::ffff:192.168.1.7]:50000\n"
        "0 0 [fe80::1]:22 [fe80::2]:50001\n"
    )
    peers = _call(PROVISION, "ssh_peer_addresses", stdin=ss)
    assert peers.split() == ["10.20.30.15", "192.168.1.7", "fe80::2"]
    assert _call(PROVISION, "peers_outside", "10.20.30.0/24", stdin=peers).split() == ["192.168.1.7", "fe80::2"]
    assert _call(PROVISION, "peers_outside", "10.20.30.0/24", "192.168.0.0/16", stdin=peers).split() == ["fe80::2"]


def test_firewall_rules_open_443_80_and_ssh_from_the_admin_subnets() -> None:
    rules = _snippet(PROVISION, "ufw_rules 1 10.1.0.0/16 10.2.0.0/24 | ufw_show_form").stdout.splitlines()
    assert rules == [
        "ufw allow 443/tcp comment 'bf-webapp-https'",
        "ufw allow 80/tcp comment 'bf-webapp-http-redirect'",
        "ufw allow from 10.1.0.0/16 to any port 22 proto tcp comment 'bf-webapp-ssh-admin'",
        "ufw allow from 10.2.0.0/24 to any port 22 proto tcp comment 'bf-webapp-ssh-admin'",
    ]
    no_http = _snippet(PROVISION, "ufw_rules 0 10.1.0.0/16 | cut -f1").stdout.splitlines()
    assert no_http == ["allow 443/tcp", "allow from 10.1.0.0/16 to any port 22 proto tcp"]


def test_firewall_defaults_hold_with_ip_forwarding_off(tmp_path: Path) -> None:
    """ufw prints ``disabled (routed)`` while net.ipv4.ip_forward is 0: a second run must still find no change."""
    status = "Status: active\nLogging: on (low)\nDefault: deny (incoming), allow (outgoing), {routed} (routed)\n"
    status += "New profiles: skip\n"
    drop, accept = tmp_path / "ufw_drop", tmp_path / "ufw_accept"
    drop.write_bytes(b'IPV6=yes\nDEFAULT_INPUT_POLICY="DROP"\nDEFAULT_FORWARD_POLICY="DROP"\n')
    accept.write_bytes(b'IPV6=yes\nDEFAULT_INPUT_POLICY="DROP"\nDEFAULT_FORWARD_POLICY="ACCEPT"\n')
    body = _function_body(PROVISION.read_text(encoding="utf-8"), "step_firewall")
    assert 'ufw_defaults_ok "$UFW_DEFAULTS"' in body and "deny (routed)'" not in body
    assert _snippet(PROVISION, 'printf "%s" "$UFW_DEFAULTS"').stdout == "/etc/default/ufw"

    def ok(routed: str, defaults: Path, text: str = status) -> bool:
        return _status(PROVISION, "ufw_defaults_ok", defaults.as_posix(), stdin=text.format(routed=routed)) == 0

    assert ok("disabled", drop)  # stock JetPack 6 without Docker: forwarding off
    assert ok("deny", drop)  # forwarding on (Docker)
    assert not ok("disabled", accept)  # routing would be allowed once forwarding is switched on
    assert not ok("allow", drop)
    assert not ok("deny", drop, status.replace("deny (incoming)", "allow (incoming)"))
    assert not ok("deny", drop, status.replace("allow (outgoing)", "deny (outgoing)"))
    assert not ok("deny", tmp_path / "missing")


# ---- verify_base.sh -------------------------------------------------------------------------------------------------

TRACKING = """Reference ID    : 0A141E01 (10.20.30.1)
Stratum         : 3
Ref time (UTC)  : Tue Oct 27 10:00:00 2026
System time     : 0.000123456 seconds slow of NTP time
Last offset     : -0.000012345 seconds
RMS offset      : 0.000045678 seconds
Frequency       : 1.234 ppm fast
Residual freq   : -0.001 ppm
Skew            : 0.012 ppm
Root delay      : 0.001234567 seconds
Root dispersion : 0.000456789 seconds
Update interval : 64.2 seconds
Leap status     : Normal
"""

UFW_STATUS = """Status: active

To                         Action      From
--                         ------      ----
443/tcp                    ALLOW       Anywhere                   # bf-webapp-https
80/tcp                     ALLOW       Anywhere                   # bf-webapp-http-redirect
22/tcp                     ALLOW       10.20.30.0/24              # bf-webapp-ssh-admin
443/tcp (v6)               ALLOW       Anywhere (v6)              # bf-webapp-https
80/tcp (v6)                ALLOW       Anywhere (v6)              # bf-webapp-http-redirect

"""

SS = """udp   UNCONN 0      0            0.0.0.0:5353       0.0.0.0:*    users:(("avahi-daemon",pid=612,fd=12))
udp   UNCONN 0      0      127.0.0.53%lo:53         0.0.0.0:*    users:(("systemd-resolve",pid=500,fd=13))
udp   UNCONN 0      0              [::1]:323           [::]:*    users:(("chronyd",pid=700,fd=6))
tcp   LISTEN 0      128          0.0.0.0:22         0.0.0.0:*    users:(("sshd",pid=800,fd=3))
tcp   LISTEN 0      128             [::]:22            [::]:*    users:(("sshd",pid=800,fd=4))
tcp   LISTEN 0      4096               *:443              *:*    users:(("caddy",pid=900,fd=7))
tcp   LISTEN 0      2048       127.0.0.1:8100       0.0.0.0:*    users:(("uvicorn",pid=950,fd=9))
tcp   LISTEN 0      128          0.0.0.0:8501       0.0.0.0:*    users:(("streamlit",pid=960,fd=6))
"""


def test_clock_offset_and_synchronisation_from_chronyc_tracking() -> None:
    assert _call(VERIFY, "tracking_offset_ms", stdin=TRACKING) == "0.123\n"
    assert _call(VERIFY, "tracking_error_bound_ms", stdin=TRACKING) == "1.198\n"  # 0.123 + 0.457 + 1.235 / 2
    assert _call(VERIFY, "tracking_value", "Reference ID", stdin=TRACKING) == "0A141E01 (10.20.30.1)\n"
    assert _status(VERIFY, "tracking_synced", stdin=TRACKING) == 0
    fast = TRACKING.replace("0.000123456 seconds slow", "0.250000000 seconds fast")
    assert _call(VERIFY, "tracking_offset_ms", stdin=fast) == "250.000\n"
    unsynced = TRACKING.replace("0A141E01 (10.20.30.1)", "00000000 ()").replace("Normal", "Not synchronised")
    assert _status(VERIFY, "tracking_synced", stdin=unsynced) == 1
    local_clock = TRACKING.replace("0A141E01 (10.20.30.1)", "7F7F0101 ()")
    assert _status(VERIFY, "tracking_synced", stdin=local_clock) == 1


def test_firewall_check_accepts_the_provisioned_rules_only() -> None:
    table = _call(VERIFY, "ufw_rules_table", stdin=UFW_STATUS)
    assert table.splitlines() == ["443/tcp\tALLOW\tAnywhere", "80/tcp\tALLOW\tAnywhere", "22/tcp\tALLOW\t10.20.30.0/24"]

    def problems(rules: str, *args: str) -> list[str]:
        return _call(VERIFY, "firewall_problems", *args, stdin=rules).splitlines()

    assert problems(table, "1") == []
    assert problems(table, "1", "10.20.30.0/24") == []
    assert problems(table, "0") == ["80/tcp is open from Anywhere (--no-http)"]
    assert problems(table, "1", "10.1.0.0/16") == [
        "22/tcp is open from 10.20.30.0/24, not an admin subnet",
        "no SSH rule for an admin subnet",
    ]
    ssh_anywhere = table.replace("10.20.30.0/24", "Anywhere") + "8501/tcp\tALLOW\tAnywhere\n"
    assert problems(ssh_anywhere, "1") == [
        "22/tcp (SSH) is open from anywhere",
        "unexpected rule: 8501/tcp ALLOW from Anywhere",
        "no SSH rule for an admin subnet",
    ]
    assert problems("80/tcp\tALLOW\tAnywhere\n", "1") == ["443/tcp is not open", "no SSH rule for an admin subnet"]


def test_open_port_list_separates_loopback_from_the_network() -> None:
    table = _call(VERIFY, "listeners_table", stdin=SS)
    rows = [line.split("\t") for line in table.splitlines()]
    assert ["udp", "53", "loopback", "127.0.0.53", "systemd-resolve"] in rows
    assert ["udp", "323", "loopback", "::1", "chronyd"] in rows
    assert ["tcp", "22", "exposed", "::", "sshd"] in rows and ["tcp", "443", "exposed", "*", "caddy"] in rows
    assert _call(VERIFY, "listeners_summary", "exposed", stdin=table).splitlines() == [
        "tcp/22 sshd",
        "tcp/443 caddy",
        "udp/5353 avahi-daemon",
        "tcp/8501 streamlit",
    ]
    assert _call(VERIFY, "listeners_summary", "loopback", stdin=table).splitlines() == [
        "udp/53 systemd-resolve",
        "udp/323 chronyd",
        "tcp/8100 uvicorn",
    ]
    assert _call(VERIFY, "unexpected_exposed", stdin=table) == "tcp/8501 streamlit\n"


def test_verify_checks_every_area_and_validates_its_arguments() -> None:
    text = VERIFY.read_text(encoding="utf-8")
    checks = re.findall(r"^\t(check_\w+)$", _function_body(text, "main"), re.M)
    assert checks == [
        "check_platform",
        "check_clock",
        "check_nvme",
        "check_power",
        "check_layout",
        "check_logs",
        "check_firewall",
        "check_ports",
        "check_units",
        "check_reboot",  # last: records the boot only when every other check passed
    ]
    assert "chronyc tracking" in _function_body(text, "check_clock")
    assert "is-system-running" in _function_body(text, "check_units")
    units = _snippet(VERIFY, 'printf "%s\\n" "${BASE_UNITS[@]}"').stdout.split()
    assert units == ["ssh.service", "chrony.service", "ufw.service", "bf-jetson-power.service", "logrotate.timer"]
    assert _snippet(PROVISION, 'printf "%s" "$POWER_UNIT"').stdout == units[3]
    assert _bash(VERIFY.as_posix(), "--help").returncode == 0
    done = _bash(VERIFY.as_posix(), "--max-offset-ms", "fast")
    assert done.returncode == 2 and "a number of milliseconds" in done.stderr
    code = 'parse_args "$@"; printf "%s|%s|%s|%s" "$MAX_OFFSET_MS" "${ADMIN_SUBNETS[*]}" '
    code += '"${EXTRA_UNITS[*]}" "$HTTP_REDIRECT"'
    done = _snippet(VERIFY, code, "--max-offset-ms", "50", "--admin-subnet", "10.1.0.0/16", "--unit", "bf-api.service")
    assert done.stdout == "50|10.1.0.0/16|bf-api.service|1"


# ---- install_python.sh (SV-6.2) -------------------------------------------------------------------------------------


def test_install_python_runs_its_steps_in_order_and_the_release_last() -> None:
    text = INSTALL.read_text(encoding="utf-8")
    main = _function_body(text, "main")
    assert re.findall(r"^\t(step_\w+)$", main, re.M) == [
        "step_platform",
        "step_packages",  # build-essential before uv sync builds quickjs
        "step_uv",
        "step_python",
        "step_env",
    ]
    assert main.index("step_env") < main.index("|| step_release") < main.index("summary")
    body = {name: _function_body(text, name) for name in ("step_platform", "step_uv", "step_python", "step_release")}
    assert "# R36 " in body["step_platform"] and "22.04" in body["step_platform"]
    assert "checksum_ok" in body["step_uv"] and "trap - RETURN" in body["step_uv"]  # the trap must not outlive it
    assert "managed_python" in body["step_python"] and "python install" in body["step_python"]
    managed = _function_body(text, "managed_python")  # never a project .venv: from / without VIRTUAL_ENV
    assert "cd /" in managed and "-u VIRTUAL_ENV" in managed and "UV_PYTHON_PREFERENCE=only-managed" in managed
    assert "sync --frozen" in body["step_release"] and "import_check_script" in body["step_release"]
    assert "[[ $owner != root ]]" in body["step_release"]
    packages = _snippet(INSTALL, 'printf "%s\\n" "${PACKAGES[@]}"').stdout.split()
    assert "build-essential" in packages


def test_install_python_argument_errors_stop_before_the_first_step() -> None:
    assert _bash(INSTALL.as_posix(), "--help").returncode == 0
    cases = {
        "--uv-version: not a version": ["--uv-version", "latest"],
        "--python: a minor version": ["--python", "3"],
        "--release: not a directory": ["--release", "/nonexistent/bf-webapp"],
        "unknown argument": ["--yes"],
        "needs a value": ["--release"],
    }
    for message, args in cases.items():
        done = _bash(INSTALL.as_posix(), *args)
        assert done.returncode == 2, (message, done.stdout, done.stderr)
        assert message in done.stderr and "Usage:" in done.stderr and "==" not in done.stdout
    code = 'parse_args "$@"; printf "%s|%s|%s|%s" "$UV_VERSION" "$PYTHON_MINOR" "$RELEASE_DIR" "$CHECK"'
    assert _snippet(INSTALL, code).stdout == "0.7.13|3.12||1"
    if _bash("-c", "echo $EUID").stdout.strip() != "0":
        done = _bash(INSTALL.as_posix())
        assert done.returncode == 1 and "run as root" in done.stderr and "==" not in done.stdout


def test_install_python_version_and_platform_helpers() -> None:
    assert _call(INSTALL, "version_of", stdin="uv 0.7.13 (62ed17b23 2025-06-12)\n") == "0.7.13\n"
    assert _call(INSTALL, "version_of", stdin="Python 3.12.10\n") == "3.12.10\n"
    assert _status(INSTALL, "version_of", stdin="error: no interpreter\n") == 1
    assert _status(INSTALL, "same_minor", "3.12.10", "3.12") == 0
    assert _status(INSTALL, "same_minor", "3.120.1", "3.12") == 1
    assert _status(INSTALL, "same_minor", "3.13.0", "3.12") == 1
    ldd = "ldd (Ubuntu GLIBC 2.35-0ubuntu3.8) 2.35\nCopyright (C) 2022 Free Software Foundation, Inc.\n"
    assert _call(INSTALL, "glibc_version", stdin=ldd) == "2.35\n"
    assert _call(INSTALL, "uv_asset", "aarch64") == "uv-aarch64-unknown-linux-gnu.tar.gz\n"
    assert _status(INSTALL, "uv_asset", "riscv64") == 1
    url = _call(INSTALL, "uv_release_url", "0.7.13", "uv-aarch64-unknown-linux-gnu.tar.gz")
    assert url == "https://github.com/astral-sh/uv/releases/download/0.7.13/uv-aarch64-unknown-linux-gnu.tar.gz\n"
    assert _status(INSTALL, "is_version", "0.7.13") == 0 and _status(INSTALL, "is_version", "0.7.x") == 1


def test_uv_download_is_checked_against_its_sha256_file(tmp_path: Path) -> None:
    tarball = tmp_path / "uv.tar.gz"
    tarball.write_bytes(b"uv release")
    real = _snippet(INSTALL, 'sha256sum "$1" | cut -d" " -f1', tarball.as_posix()).stdout.strip()
    assert len(real) == 64
    other = ("0" if real[0] != "0" else "1") + real[1:]
    assert _status(INSTALL, "checksum_ok", tarball.as_posix(), stdin=f"{real.upper()} *uv.tar.gz\n") == 0
    assert _status(INSTALL, "checksum_ok", tarball.as_posix(), stdin=f"{other}  uv.tar.gz\n") == 1
    assert _status(INSTALL, "checksum_ok", tarball.as_posix(), stdin="") == 1


def test_uv_env_pins_the_managed_python_and_the_frozen_lock() -> None:
    env = _call(INSTALL, "render_uv_env", "/opt/bf-webapp/python", "/var/lib/bf-webapp/cache/uv", "3.12")
    values = dict(line.split("=", 1) for line in env.splitlines() if line and not line.startswith("#"))
    assert values == {
        "UV_PYTHON_INSTALL_DIR": "/opt/bf-webapp/python",
        "UV_PYTHON": "3.12",
        "UV_PYTHON_PREFERENCE": "only-managed",
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_CACHE_DIR": "/var/lib/bf-webapp/cache/uv",
        "UV_LINK_MODE": "copy",
        "UV_FROZEN": "1",
        "UV_NO_PROGRESS": "1",
    }
    assert _snippet(INSTALL, 'printf "%s %s" "$UV_ENV_FILE" "$UV_ENV_FILE_MODE"').stdout == "/etc/bf-webapp/uv.env 640"
    bfapp = _call(INSTALL, "uv_env_for", "bfapp", stdin=env).splitlines()
    operator = _call(INSTALL, "uv_env_for", "operator", stdin=env).splitlines()
    assert len(bfapp) == 8 and "UV_CACHE_DIR=/var/lib/bf-webapp/cache/uv" in bfapp
    assert operator == [line for line in bfapp if not line.startswith("UV_CACHE_DIR=")]  # keeps their own cache


def test_import_check_covers_the_audited_packages() -> None:
    modules = _snippet(INSTALL, 'printf "%s\\n" "${AUDIT_MODULES[@]}"').stdout.split()
    script = _call(INSTALL, "import_check_script")
    compile(script, "import_check", "exec")
    assert "modules = [" + ", ".join(f'"{m}"' for m in modules) + "]" in script
    assert 'method="highs"' in script and "sys.exit(1 if failed else 0)" in script
    for name in ("torch", "scipy", "psycopg2", "duckdb", "chromadb", "pyarrow", "sentence_transformers", "quickjs"):
        assert name in modules  # docs/jetson_wheel_audit.md: highspy is scipy's HiGHS, checked by the LP
    readme = (JETSON / "README.md").read_text(encoding="utf-8")
    listed = re.search(r"then imports (.*?) in the synced", readme)
    assert listed and set(re.split(r", | and ", listed.group(1))) == set(modules)


# ---- benchmark.sh (SV-6.2) ------------------------------------------------------------------------------------------

# real output of scripts/optimiser_bench.py (laptop, 29 Oct 2026)
CASES = "cases built in 2.9 s: yard reference, calibration fallback (live, 2026-07-01 to 2026-10-07), cpus 8, solver "
CASES += "processes 8\n"
SOLVE_LOG = CASES + (
    "one network solve: 1.64 s the first in this process (it builds the STRATA-Flow context), then 1.35-1.70 s\n"
)
JACOBIAN_LOG = CASES + (
    "network Jacobian (17 solves): serial 28.3 s; 8 solver processes started in 12.2 s (0 with the STRATA-Flow "
    "context from the forkserver), then the Jacobian 14.7 s\n"
)
STEP2_LOG = CASES + (
    "network Jacobian: cache jacobian_6e45192249a45d30.json\n"
    "[parallel] step2 strata:  179.5 s, 90 network solves (450.9 s solving); 2270 t/d, coke 300.31 kg/tHM, cost "
    "29,834.6 Rs/tHM\n"
    "[parallel] step2  fixed:    2.1 s, 0 network solves (0.0 s solving); 2270 t/d, coke 302.44 kg/tHM, cost "
    "29,833.9 Rs/tHM\n"
    "[parallel] wall time of the solves 181.4 s\n"
    "above the limit or different\n"
)
MODES = ["solve", "jacobian", "step2", "max_production", "step1"]
BENCH_DOC = ROOT / "docs" / "jetson_benchmark.md"
START, END = "<!-- benchmark:start -->", "<!-- benchmark:end -->"


def test_benchmark_targets_are_section_10_of_the_plan() -> None:
    targets = {mode: _call(BENCH, "target_seconds", mode).strip() for mode in MODES}
    assert targets == {"solve": "2.8", "jacobian": "6", "step2": "90", "max_production": "300", "step1": "600"}
    assert _snippet(BENCH, 'printf "%s " "${MODES_ALL[@]}"').stdout.split() == MODES
    assert _call(BENCH, "bench_args", "jacobian", "8") == "--mode jacobian --solvers 8 --limit-s 6\n"
    assert _call(BENCH, "bench_args", "step1", "8") == "--mode step1 --limit-s 600\n"
    assert _status(BENCH, "target_seconds", "step3") == 1


def test_bench_output_parsers_read_the_real_output() -> None:
    assert _call(BENCH, "bench_seconds", "solve", stdin=SOLVE_LOG) == "1.70\n"  # the slowest repeat
    note = _call(BENCH, "bench_note", "solve", stdin=SOLVE_LOG)
    assert note == "first solve 1.64 s (builds the context), then 1.35-1.70 s\n"
    assert _call(BENCH, "bench_seconds", "jacobian", stdin=JACOBIAN_LOG) == "14.7\n"
    assert (
        _call(BENCH, "bench_note", "jacobian", stdin=JACOBIAN_LOG) == "serial 28.3 s; 8 processes started in 12.2 s\n"
    )
    assert _call(BENCH, "bench_seconds", "step2", stdin=STEP2_LOG) == "181.4\n"
    assert _call(BENCH, "bench_note", "step2", stdin=STEP2_LOG) == "90 network solves in 2 case(s)\n"
    crashed = "cases built in 2.3 s\nTraceback (most recent call last):\n"
    for mode in MODES:
        assert _status(BENCH, "bench_seconds", mode, stdin=crashed) == 1


def test_results_block_and_its_place_in_the_document(tmp_path: Path) -> None:
    assert _status(BENCH, "above_target", "14.7", "6") == 0 and _status(BENCH, "above_target", "6", "6") == 1
    assert _call(BENCH, "fmt_seconds", "2.8") == "2.8 s\n"
    assert _call(BENCH, "fmt_seconds", "1158.2") == "1158 s (19.3 min)\n"
    text = BENCH_DOC.read_text(encoding="utf-8")
    pending = "".join(f"{m}\t-\t{_call(BENCH, 'target_seconds', m).strip()}\tpending: device\t\n" for m in MODES)
    assert _call(BENCH, "results_rows", stdin=pending) in text  # the document's pending table is the script's own
    tsv = (
        "solve\t1.70\t2.8\tPASS within target\tfirst solve 1.64 s (builds the context), then 1.35-1.70 s\n"
        "step1\t-\t600\tFAIL no result (exit 1)\t\n"
    )
    block = _call(BENCH, "results_block", "2026-11-02 08:00Z on NVIDIA Jetson AGX Orin", stdin=tsv)
    assert block.splitlines() == [
        START,
        "Measured 2026-11-02 08:00Z on NVIDIA Jetson AGX Orin",
        "",
        "| Operation | Target (AGX Orin) | Measured | Status | Notes |",
        "|---|---|---|---|---|",
        "| One STRATA-Flow network solve (worst of 3 repeats after the first) | 2.8 s | 1.7 s | PASS within target | "
        "first solve 1.64 s (builds the context), then 1.35-1.70 s |",
        "| Step 1, cheapest production level and frontier (STRATA-ω) | 600 s (10.0 min) | — | FAIL no result (exit 1) "
        "| — |",
        END,
    ]
    copy = tmp_path / "doc.md"
    copy.write_bytes(text.encode())
    replaced = _call(BENCH, "replace_between", copy.as_posix(), stdin=block)
    head, tail = text[: text.index(START)], text[text.index(END) + len(END) + 1 :]
    assert replaced == head + block + tail
    copy.write_bytes(b"no markers\n")
    assert _status(BENCH, "replace_between", copy.as_posix(), stdin=block) == 1


def test_issue_for_a_runtime_above_target() -> None:
    repo = _call(BENCH, "repo_of_url", "git@github.com:saikumar-voptimai/evonith_bf_webapp.git")
    assert repo == "saikumar-voptimai/evonith_bf_webapp\n"
    assert _call(BENCH, "repo_of_url", "https://github.com/owner/name") == "owner/name\n"
    assert _call(BENCH, "repo_of_url", "https://gitlab.com/owner/name.git") == ""
    title = _call(BENCH, "issue_title", "step1", "1158.2", "600")
    assert title == "Jetson benchmark: step1 1158 s (19.3 min) above the 600 s (10.0 min) target (SV-6.2)\n"
    body = _call(
        BENCH, "issue_body", "step1", "1158.2", "600", "host", "1260 network solves in 1 case(s)", "/tmp/x.log"
    )
    assert "**1158 s (19.3 min)**" in body and "Refs #316" in body and "--mode step1 --compare" in body


def test_benchmark_argument_errors_and_defaults() -> None:
    assert _bash(BENCH.as_posix(), "--help").returncode == 0
    cases = {
        "--mode: one of": ["--mode", "step3"],
        "--solvers: a number": ["--solvers", "eight"],
        "--release: not a directory": ["--release", "/nonexistent/bf"],
        "unknown argument": ["--fast"],
        "needs a value": ["--doc"],
    }
    for message, args in cases.items():
        done = _bash(BENCH.as_posix(), *args)
        assert done.returncode == 2 and message in done.stderr and "==" not in done.stdout, (message, done.stderr)
    code = 'parse_args "$@"; printf "%s|%s|%s|%s|%s" "${MODES[*]}" "$SOLVERS" "$DOC" "$LIVE_DAY" "$OPEN_ISSUE"'
    assert _snippet(BENCH, code).stdout == f"{' '.join(MODES)}|8|docs/jetson_benchmark.md|0|0"


# fake uv: canned optimiser_bench.py output per mode; the jacobian above its target, step1 crashes
FAKE_UV = r"""#!/usr/bin/env bash
case "$*" in
--version) echo "uv 0.7.13 (62ed17b23 2025-06-12)"; exit 0 ;;
"run --no-sync python --version") echo "Python 3.12.10"; exit 0 ;;
esac
printf '%s\n' "$*" >>"$BF_FAKE_CALLS"
mode=$(sed -nE 's/.*--mode ([a-z_0-9]+).*/\1/p' <<<"$*")
case $mode in
solve) printf '%s' "$SOLVE_LOG" ;;
jacobian) printf '%s' "$JACOBIAN_LOG"; exit 1 ;;
step2) printf '[parallel] step2 strata:   61.0 s, 73 network solves (90.0 s solving); x\n'
	printf '[parallel] wall time of the solves 61.5 s\n' ;;
max_production) printf '[parallel] max_production strata:  250.0 s, 347 network solves (400.0 s solving); x\n'
	printf '[parallel] wall time of the solves 251.0 s\n' ;;
step1) echo "Traceback (most recent call last):"; exit 1 ;;
esac
"""


def test_benchmark_end_to_end_on_a_fake_release(tmp_path: Path) -> None:
    release = tmp_path / "release"
    for path, content in {
        "scripts/optimiser_bench.py": b"# fake\n",
        "uv.lock": b"version = 1\n",
        ".venv/bin/python": b"#!/bin/sh\n",
        "docs/jetson_benchmark.md": BENCH_DOC.read_bytes(),
    }.items():
        (release / path).parent.mkdir(parents=True, exist_ok=True)
        (release / path).write_bytes(content)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "uv").write_bytes(FAKE_UV.encode())
    calls = tmp_path / "calls.txt"
    code = (
        'chmod +x "$2/uv" "$1/.venv/bin/python"; bin=$(cygpath -u "$2" 2>/dev/null || printf %s "$2"); '
        'export PATH="$bin:$PATH" BF_FAKE_CALLS="$3" SOLVE_LOG="$4" '
        'JACOBIAN_LOG="$5"; "$6" --release "$1" --log-dir "$1/logs"'
    )
    args = [release.as_posix(), (tmp_path / "bin").as_posix(), calls.as_posix(), SOLVE_LOG, JACOBIAN_LOG]
    done = _bash("-c", code, "bash", *args, BENCH.as_posix())
    assert done.returncode == 1, done.stdout + done.stderr  # jacobian above target, step1 without result
    made = calls.read_text(encoding="utf-8").splitlines()
    assert [line.split("--mode ")[1].split()[0] for line in made] == MODES
    assert made[1] == "run --no-sync python -u scripts/optimiser_bench.py --mode jacobian --solvers 8 --limit-s 6"
    text = (release / "docs" / "jetson_benchmark.md").read_text(encoding="utf-8")
    block = text[text.index(START) : text.index(END)]
    assert "Measured " in block and "uv 0.7.13, Python 3.12.10" in block and "pending: device" not in block
    assert "| 2.8 s | 1.7 s | PASS within target |" in block
    assert "| 6.0 s | 14.7 s | FAIL above target | serial 28.3 s; 8 processes started in 12.2 s |" in block
    assert "| 90.0 s | 61.5 s | PASS within target | 73 network solves in 1 case(s) |" in block
    assert "| 300 s (5.0 min) | 251 s (4.2 min) | PASS within target |" in block
    assert "| 600 s (10.0 min) | — | FAIL no result (exit 1) | — |" in block
    original = BENCH_DOC.read_text(encoding="utf-8")
    assert text.startswith(original[: original.index(START)]) and text.endswith(original[original.index(END) :])
    # one issue command per runtime above target (a mode without result fails the run but has no runtime to report)
    commands = [line for line in done.stdout.splitlines() if "gh issue create" in line]
    assert len(commands) == 1 and "--repo saikumar-voptimai/evonith_bf_webapp" in commands[0]
    assert "Jetson benchmark: jacobian 14.7 s above the 6.0 s target (SV-6.2)" in commands[0]
    issue = (release / "logs" / "issue-jacobian.md").read_text(encoding="utf-8")
    assert "Refs #316" in issue and "Bench: serial 28.3 s" in issue and "logs/jacobian.log`" in issue
