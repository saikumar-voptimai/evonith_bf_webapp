"""Release and rollback on the Jetson (SV-6.4, #318): ``deploy/jetson/release.sh`` and ``rollback.sh``.

The pure helpers run in bash after sourcing the script. The update and rollback run end to end on a fake host: the
paths of the script point into a temporary directory (``OPT_DIR``, ``DATA_DIR``, ``ETC_DIR``, ...), and ``gh``,
``systemctl``, ``curl``, ``runuser`` and ``chown`` are stubs. The ``curl`` stub answers as Caddy would for the release
``current`` points at; a release named in ``BF_FAKE_BROKEN`` answers 502 on ``/api``, which is the failed smoke test
that must roll back without a manual step. Each fake release is a real ``git archive``-like tarball (pax header with
the commit) with its CI artefacts and ``SHA256SUMS``; its ``install_python.sh`` writes a fake ``.venv/bin/python``
that records the alembic calls. Needs bash with symbolic links (Git for Windows with Developer Mode on Windows).
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from services.release import ReleaseInfo, read_release
from tests.test_jetson_scripts import BASH, _function_body

ROOT = Path(__file__).resolve().parents[1]
JETSON = ROOT / "deploy" / "jetson"
RELEASE = JETSON / "release.sh"
ROLLBACK = JETSON / "rollback.sh"

pytestmark = pytest.mark.requires_service(BASH is not None, "bash is not installed (Git for Windows, or BF_BASH_BIN)")

# Git for Windows makes copies for `ln -s` unless told to make native links (needs Developer Mode or admin)
BASH_ENV = {"MSYS": "winsymlinks:nativestrict"}


def _run_bash(*args: str, env: dict[str, str] | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    assert BASH is not None
    full = {**os.environ, **BASH_ENV, **(env or {})}
    done = subprocess.run([BASH, *args], capture_output=True, timeout=timeout, check=False, env=full)
    return subprocess.CompletedProcess(done.args, done.returncode, done.stdout.decode(), done.stderr.decode())


def _helper(code: str, *args: str, stdin: str = "") -> subprocess.CompletedProcess[str]:
    """``code`` in bash after sourcing release.sh (errexit off); ``$1..`` are ``args``."""
    assert BASH is not None
    script = f'source "$1"; set +e; shift; {code}'
    done = subprocess.run(
        [BASH, "-c", script, "bash", RELEASE.as_posix(), *args],
        input=stdin.encode(),
        capture_output=True,
        timeout=60,
        check=False,
        env={**os.environ, **BASH_ENV},
    )
    return subprocess.CompletedProcess(done.args, done.returncode, done.stdout.decode(), done.stderr.decode())


# ---- the scripts ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("script", [RELEASE, ROLLBACK], ids=lambda p: p.name)
def test_script_is_valid_bash_with_lf_line_ends_and_executable(script: Path) -> None:
    raw = script.read_bytes()
    assert b"\r" not in raw and raw.startswith(b"#!/usr/bin/env bash\n")
    done = _run_bash("-n", script.as_posix())
    assert done.returncode == 0, done.stderr
    git = shutil.which("git")
    if git:
        listed = subprocess.run([git, "ls-files", "-s", script.name], cwd=JETSON, capture_output=True, text=True)
        if listed.returncode == 0 and listed.stdout:
            assert listed.stdout.startswith("100755 "), "not executable in git (git add --chmod=+x)"


def test_readme_documents_every_option_and_the_steps() -> None:
    readme = (JETSON / "README.md").read_text(encoding="utf-8")
    usage = _helper("usage").stdout
    options = set(re.findall(r"--[a-z][a-z-]+", usage)) - {"--help", "--preserve-env"}
    assert options and not sorted(o for o in options if o not in readme)
    for word in ("release.sh", "rollback.sh", "RELEASE.json", "SHA256SUMS", "/api/version", "previous"):
        assert word in readme, word


def test_update_runs_its_steps_in_order() -> None:
    text = RELEASE.read_text(encoding="utf-8")
    steps = re.findall(r"^\t+(step_\w+|activate)\b", _function_body(text, "main_update"), re.M)
    assert steps == [
        "step_fetch",
        "step_verify",
        "step_unpack",
        "step_sync",
        "step_migrate",  # before the switch: a failed migration leaves the running release alone
        "step_record",
        "activate",  # switch, restart, smoke test, automatic rollback
        "step_prune",
    ]


def test_rollback_sh_runs_release_sh_with_rollback(tmp_path: Path) -> None:
    shutil.copy(ROLLBACK, tmp_path / "rollback.sh")
    (tmp_path / "release.sh").write_bytes(b'#!/usr/bin/env bash\nprintf "%s|" "$@"\n')
    os.chmod(tmp_path / "release.sh", 0o755)
    done = _run_bash((tmp_path / "rollback.sh").as_posix(), "v1.2.0", "--smoke-timeout", "60")
    assert done.returncode == 0, done.stderr
    assert done.stdout == "--rollback|v1.2.0|--smoke-timeout|60|"


# ---- pure helpers ---------------------------------------------------------------------------------------------------


def test_tags_and_repositories() -> None:
    for tag, ok in {"v1.2.0": True, "v1.2.0-rc.1": True, "v10": True, "1.2.0": False, "v1;reboot": False,
                    "v1/../x": False, "va": False}.items():  # fmt: skip
        assert (_helper('valid_tag "$1"', tag).returncode == 0) is ok, tag
    assert _helper('valid_repo "$1"', "saikumar-voptimai/evonith_bf_webapp").returncode == 0
    assert _helper('valid_repo "$1"', "a/b/c").returncode == 1


def test_sha256sums_must_list_the_three_files() -> None:
    names = ["frontend-build.tar.gz", "uv.lock", "pnpm-lock.yaml"]
    sums = "".join(f"{c * 64}  {n}\n" for c, n in zip("abc", names, strict=True))
    code = "sums_cover frontend-build.tar.gz uv.lock pnpm-lock.yaml"
    assert _helper(code, stdin=sums).returncode == 0
    assert _helper(code, stdin=sums.replace("  uv.lock", "  uvXlock")).returncode == 1  # the dot is literal
    assert _helper(code, stdin=sums + sums.splitlines()[1] + "\n").returncode == 1  # listed twice
    assert _helper(code, stdin=sums.replace("a" * 64, "a" * 63)).returncode == 1


def test_environment_file_is_parsed_not_sourced() -> None:
    env = "\n".join([
        "# comment", "", "BF_SITE=bf02.plant", 'export BF_API_JWT_SECRET="ab cd"', "BF_HTTPS_PORT='8443'",
        "DATABASE_URL=postgresql://u:p@h/db?sslmode=require&channel_binding=require", "  not an assignment",
        "X=a=b", "Y=crlf\r",
    ])  # fmt: skip
    done = _helper("env_assignments", stdin=env)
    assert done.stdout.splitlines() == [
        "BF_SITE=bf02.plant",
        "BF_API_JWT_SECRET=ab cd",
        "BF_HTTPS_PORT=8443",
        "DATABASE_URL=postgresql://u:p@h/db?sslmode=require&channel_binding=require",
        "X=a=b",
        "Y=crlf",
    ]
    assert _helper('env_value "$1"', "X", stdin="X=1\nXY=2\nX=3\n").stdout.strip() == "3"


@pytest.mark.parametrize(
    ("env", "configured"),
    [
        ("DATABASE_URL=postgresql://h/db\n", True),
        ("BF_TWIN_DB_URL=postgresql://h/twin\n", True),
        ("BF_TWIN_DB_URL=Off\nDATABASE_URL=postgresql://h/db\n", False),  # data/twin/engine.py OFF
        ("BF_TWIN_DB_URL=false\n", False),
        ("BF_SITE=x\n", False),
    ],
)
def test_twin_database_as_the_twin_store_resolves_it(env: str, configured: bool) -> None:
    assert (_helper("twin_db_configured", stdin=env).returncode == 0) is configured


def test_alembic_revision_and_prune_list() -> None:
    out = "INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.\ntwin_0009 (head)\n"
    assert _helper("alembic_revision", stdin=out).stdout.strip() == "twin_0009"
    assert _helper("alembic_revision", stdin="INFO  [alembic] Will assume transactional DDL.\n").returncode == 1
    names = "v5\nv4\n.stage-v6\nv3\nv2\nv1\n"  # newest first
    assert _helper('prune_list 3 "$1" "$2"', "v1", "v4", stdin=names).stdout.split() == ["v2"]
    assert _helper('prune_list 2 "$1" "$2"', "v5", "v4", stdin=names).stdout.split() == ["v3", "v2", "v1"]


def test_commit_id_of_a_git_archive(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    head = subprocess.run([git, "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    archive = tmp_path / "source.tar.gz"
    with archive.open("wb") as out:
        subprocess.run([git, "archive", "--format=tar.gz", "HEAD", "--", "deploy/jetson/README.md"], cwd=ROOT,
                       stdout=out, check=True)  # fmt: skip
    done = _helper('tar_commit_id <"$1"', archive.as_posix())
    assert done.returncode == 0 and done.stdout.strip() == head.strip()
    (tmp_path / "plain.gz").write_bytes(gzip.compress(b"no header"))
    assert _helper('tar_commit_id <"$1"', (tmp_path / "plain.gz").as_posix()).returncode == 1


def test_record_written_by_the_script_is_read_by_the_twin_api(tmp_path: Path) -> None:
    commit = "0123456789abcdef0123456789abcdef01234567"
    args = ["v1.2.0", commit, "v1.2.0", "2026-10-29T17:05:00Z", "twin_0008", "twin_0009"]
    (tmp_path / "RELEASE.json").write_text(_helper('render_release_json "$@"', *args).stdout, encoding="utf-8")
    record = json.loads((tmp_path / "RELEASE.json").read_text(encoding="utf-8"))
    assert record["twin_revision_before"] == "twin_0008" and record["twin_revision_after"] == "twin_0009"
    info = read_release(tmp_path)
    assert (info.tag, info.commit, info.installed_at.isoformat()) == ("v1.2.0", commit, "2026-10-29T17:05:00+00:00")


def test_argument_errors_stop_before_any_step() -> None:
    cases = {
        "the release tag is required": [],
        "not a release tag": ["1.2.0"],
        "one tag only": ["v1.2.0", "v1.3.0"],
        "--repo: OWNER/NAME": ["v1.2.0", "--repo", "evonith"],
        "--keep: a whole number of at least 2": ["v1.2.0", "--keep", "1"],
        "--smoke-timeout: seconds": ["v1.2.0", "--smoke-timeout", "0"],
        "--artefacts: not a directory": ["v1.2.0", "--artefacts", "/nonexistent/bf"],
        "--rollback takes a tag and --smoke-timeout only": ["--rollback", "--keep", "4"],
        "unknown argument": ["v1.2.0", "--yes"],
        "needs a value": ["v1.2.0", "--repo"],
    }
    for message, args in cases.items():
        done = _run_bash(RELEASE.as_posix(), *args)
        assert done.returncode == 2, (message, done.stdout, done.stderr)
        assert message in done.stderr and "Usage:" in done.stderr and "==" not in done.stdout


def test_a_user_other_than_root_is_refused() -> None:
    if _run_bash("-c", "echo $EUID").stdout.strip() == "0":
        pytest.skip("running as root")
    done = _run_bash(RELEASE.as_posix(), "v1.2.0")
    assert done.returncode == 1 and "run as root" in done.stderr and "==" not in done.stdout


# ---- end to end on a fake host --------------------------------------------------------------------------------------

FAKE_INSTALL_PYTHON = r"""#!/usr/bin/env bash
# install_python.sh of a fake release: "uv sync" makes .venv/bin/python, which records its alembic calls
set -euo pipefail
printf 'install_python %s\n' "$*" >>"$BF_FAKE_CALLS"
[[ ${BF_FAKE_SYNC_FAIL:-0} != 1 ]] || { echo "uv sync --frozen failed" >&2; exit 1; }
dir=$2
mkdir -p "$dir/.venv/bin"
cat >"$dir/.venv/bin/python" <<PY
#!/usr/bin/env bash
printf 'python %s | DATABASE_URL=%s HOME=%s\n' "\$*" "\${DATABASE_URL:-}" "\${HOME:-}" >>"$BF_FAKE_CALLS"
case "\$*" in
*"alembic -n twin current"*)
	echo "INFO  [alembic.runtime.migration] Context impl PostgresqlImpl."
	cat "$BF_FAKE_REV" ;;
*"alembic -n twin upgrade head"*)
	[[ "${BF_FAKE_MIGRATE_FAIL:-0}" != 1 ]] || exit 1
	echo "twin_0009 (head)" >"$BF_FAKE_REV" ;;
esac
PY
chmod +x "$dir/.venv/bin/python"
"""

STUBS = {
    "gh": r"""#!/usr/bin/env bash
printf 'gh %s\n' "$*" >>"$BF_FAKE_CALLS"
[[ $1 == release && $2 == download ]] || exit 2
tag=$3; shift 3
src="$BF_FAKE_GH/$tag"
[[ -d $src ]] || { echo "release not found" >&2; exit 1; }
dir=. output="" archive="" patterns=()
while (($#)); do
	case $1 in
	--dir) dir=$2; shift 2 ;;
	--pattern) patterns+=("$2"); shift 2 ;;
	--archive) archive=$2; shift 2 ;;
	--output) output=$2; shift 2 ;;
	--repo) shift 2 ;;
	*) shift ;;
	esac
done
if [[ -n $archive ]]; then cp "$src/source.tar.gz" "$output"; exit 0; fi
for p in "${patterns[@]}"; do cp "$src/$p" "$dir/$p"; done
""",
    "systemctl": r"""#!/usr/bin/env bash
printf 'systemctl %s\n' "$*" >>"$BF_FAKE_CALLS"
exit "${BF_FAKE_SYSTEMCTL_RC:-0}"
""",
    "runuser": r"""#!/usr/bin/env bash
printf 'runuser %s\n' "$2" >>"$BF_FAKE_CALLS"
while (($#)) && [[ $1 != -- ]]; do shift; done
shift
exec "$@"
""",
    "chown": "#!/usr/bin/env bash\nexit 0\n",
    # Caddy in front of the release that `current` points at
    "curl": r"""#!/usr/bin/env bash
url="" resolve=""
while (($#)); do
	case $1 in
	--resolve) resolve=$2; shift 2 ;;
	https://*) url=$1; shift ;;
	*) shift ;;
	esac
done
path=/${url#https://*/}
printf 'curl %s %s\n' "$resolve" "$path" >>"$BF_FAKE_CALLS"
rel="$BF_FAKE_OPT/current"
cur=$(readlink "$rel" 2>/dev/null) || { printf '\n000'; exit 7; }
tag=${cur##*/}
code=200 body=""
if [[ " ${BF_FAKE_BROKEN:-} " == *" $tag "* && $path == /api/* ]]; then
	code=502 body="bad gateway"
else
	case $path in
	/api/health) body='{"status":"ok","service":"twin-api"}' ;;
	/api/version)
		t=$(sed -n 's/.*"tag": "\(.*\)".*/\1/p' "$rel/RELEASE.json")
		c=$(sed -n 's/.*"commit": "\(.*\)".*/\1/p' "$rel/RELEASE.json")
		body="{\"version\":\"0.1.0\",\"git_sha\":\"$c\",\"environment\":\"prod\",\"release\":\"$t\"}" ;;
	/api/health/deep) code=404 body='{"title":"Not Found"}' ;;
	/legacy/_stcore/health) body=ok ;;
	/) body=$(cat "$rel/frontend/build/index.html") ;;
	/_app/version.json) body=$(cat "$rel/frontend/build/_app/version.json") ;;
	*) code=404 ;;
	esac
fi
printf '%s\n%s' "$body" "$code"
""",
}

RUN_MAIN = r"""
p() { cygpath -u "$1" 2>/dev/null || printf '%s' "$1"; }
source "$(p "$1")"
OPT_DIR=$(p "$2") DATA_DIR=$(p "$3") ETC_DIR=$(p "$4") UNIT_DIR=$(p "$5") LOCK_DIR="$(p "$6")/bf-webapp-release"
export PATH="$(p "$7"):$PATH" BF_FAKE_OPT=$OPT_DIR BF_FAKE_GH=$(p "$8") BF_FAKE_CALLS=$(p "$9") BF_FAKE_REV=$(p "${10}")
SMOKE_RETRY_S=0.2
require_root() { :; }
if [[ ${BF_FAKE_LOCKED:-0} == 1 ]]; then mkdir -p "$LOCK_DIR" && printf '%s\n' "$$" >"$LOCK_DIR/pid"; fi
shift 10
main "$@"
"""


def _tar_gz(entries: dict[str, tuple[bytes, int]], pax: dict[str, str] | None = None) -> bytes:
    raw = io.BytesIO()
    fmt = tarfile.PAX_FORMAT if pax else tarfile.GNU_FORMAT
    with tarfile.open(fileobj=raw, mode="w", format=fmt, pax_headers=pax or {}) as tar:
        for name, (data, mode) in sorted(entries.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), mode, 1_761_700_000
            tar.addfile(info, io.BytesIO(data))
    return gzip.compress(raw.getvalue(), mtime=0)


@dataclass
class Host:
    """A fake Jetson: the script's paths, the stubs and the GitHub releases, all under one temporary directory."""

    root: Path

    def __post_init__(self) -> None:
        for d in ("opt/releases", "data/storage", "etc", "units", "lock", "bin", "gh"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        env = [
            "BF_API_ENVIRONMENT=prod",
            'BF_SITE="bf02.test"',
            "BF_HTTPS_PORT=8443",
            "DATABASE_URL=postgresql://u:p@db.neon.tech/bf?sslmode=require&channel_binding=require",
        ]
        (self.root / "etc" / "env").write_text("\n".join(env) + "\n", encoding="utf-8")
        (self.root / "etc" / "uv.env").write_text("UV_FROZEN=1\n", encoding="utf-8")
        for name, body in STUBS.items():
            (self.root / "bin" / name).write_bytes(body.encode())
            os.chmod(self.root / "bin" / name, 0o755)
        (self.root / "rev").write_text("twin_0008 (head)\n", encoding="utf-8")
        (self.root / "calls").write_text("", encoding="utf-8")
        (self.root / "units" / "bf-twin-api.service").write_text("[Unit]\n", encoding="utf-8")

    @property
    def opt(self) -> Path:
        return self.root / "opt"

    @property
    def storage(self) -> Path:
        return self.root / "data" / "storage"

    def publish(self, tag: str, *, tamper: str | None = None, lock_mismatch: bool = False) -> str:
        """A GitHub release of ``tag`` made by release.yml, and its source archive; returns the commit."""
        commit = hashlib.sha1(tag.encode()).hexdigest()
        lock = f"version = 1  # {tag}\n".encode()
        pnpm = f"lockfileVersion: '9.0'  # {tag}\n".encode()
        top = f"evonith_bf_webapp-{tag.removeprefix('v')}/"
        source = {
            "pyproject.toml": (b"[project]\nname = 'bf-webapp'\n", 0o644),
            "uv.lock": (lock + (b"# edited\n" if lock_mismatch else b""), 0o644),
            "frontend/pnpm-lock.yaml": (pnpm, 0o644),
            "alembic.ini": (b"[twin]\n", 0o644),
            "src/api/main.py": (b"app = None\n", 0o644),
            "src/storage/.gitkeep": (b"", 0o644),
            "src/storage/furnacemind/SKILLS.md": (f"skills of {tag}\n".encode(), 0o644),
            "deploy/jetson/install_python.sh": (FAKE_INSTALL_PYTHON.encode(), 0o755),
            "deploy/jetson/systemd/bf-twin-api.service": (f"[Unit]\n# {tag}\n".encode(), 0o644),
        }
        build = {
            "build/200.html": (b"<!doctype html><html><body>BF-02</body></html>", 0o644),
            "build/index.html": (b"<!doctype html><html><body>BF-02</body></html>", 0o644),
            "build/_app/version.json": (json.dumps({"version": tag}).encode(), 0o644),
        }
        out = self.root / "gh" / tag
        out.mkdir()
        (out / "source.tar.gz").write_bytes(_tar_gz({top + k: v for k, v in source.items()}, {"comment": commit}))
        (out / "frontend-build.tar.gz").write_bytes(_tar_gz(build))
        (out / "uv.lock").write_bytes(lock)
        (out / "pnpm-lock.yaml").write_bytes(pnpm)
        names = ["frontend-build.tar.gz", "uv.lock", "pnpm-lock.yaml"]
        sums = "".join(f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}\n" for n in names)
        (out / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")
        if tamper:
            (out / tamper).write_bytes((out / tamper).read_bytes() + b"x")
        return commit

    def run(self, *args: str, **fake: str) -> subprocess.CompletedProcess[str]:
        (self.root / "calls").write_text("", encoding="utf-8")
        env = {f"BF_FAKE_{k.upper()}": v for k, v in fake.items()}
        paths = [RELEASE, self.opt, self.root / "data", self.root / "etc", self.root / "units", self.root / "lock"]
        paths += [self.root / "bin", self.root / "gh", self.root / "calls", self.root / "rev"]
        return _run_bash("-c", RUN_MAIN, "bash", *(p.as_posix() for p in paths), *args, env=env)

    def calls(self) -> list[str]:
        return (self.root / "calls").read_text(encoding="utf-8").splitlines()

    def link(self, name: str) -> str | None:
        path = self.opt / name
        return Path(os.readlink(path)).name if path.is_symlink() else None

    def releases(self) -> list[str]:
        return sorted(p.name for p in (self.opt / "releases").iterdir())

    def log(self) -> list[str]:
        path = self.opt / "release.log"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


@pytest.fixture()
def host(tmp_path: Path) -> Host:
    probe = _run_bash("-c", 'cd "$1" && mkdir d && ln -s d l && [[ -L l ]]', "bash", tmp_path.as_posix())
    if probe.returncode != 0:
        pytest.skip("bash cannot make symbolic links here (Windows: enable Developer Mode)")
    shutil.rmtree(tmp_path / "d")
    (tmp_path / "l").unlink()
    return Host(tmp_path / "host")


def _out(done: subprocess.CompletedProcess[str]) -> str:
    return done.stdout + done.stderr


def test_update_installs_switches_restarts_and_smoke_tests(host: Host) -> None:
    commit = host.publish("v1.0.0")
    done = host.run("v1.0.0", "--smoke-timeout", "5")
    assert done.returncode == 0, _out(done)
    assert (host.link("current"), host.link("previous")) == ("v1.0.0", None)
    rel = host.opt / "releases" / "v1.0.0"
    assert read_release(rel) == ReleaseInfo(
        tag="v1.0.0", commit=commit, installed_at=read_release(rel).installed_at, frontend_build="v1.0.0"
    )
    record = json.loads((rel / "RELEASE.json").read_text(encoding="utf-8"))
    assert (record["twin_revision_before"], record["twin_revision_after"]) == ("twin_0008", "twin_0009")
    # application state on the data disk, seeded once from the release
    assert (rel / "src" / "storage").is_symlink() and (host.storage / "furnacemind" / "SKILLS.md").is_file()
    calls = host.calls()
    assert calls[0].startswith("gh release download v1.0.0 --repo saikumar-voptimai/evonith_bf_webapp")
    assert "--archive tar.gz" in calls[1]
    assert calls[2] == f"install_python --release {_posix(rel)}"
    alembic = [c for c in calls if c.startswith("python ")]
    assert [c.split(" | ")[0] for c in alembic] == [
        "python -m alembic -n twin current",
        "python -m alembic -n twin upgrade head",
        "python -m alembic -n twin current",
    ]
    # the env file is passed as it is (the & of the URL), to bfapp, and nothing else from root's environment
    assert "DATABASE_URL=postgresql://u:p@db.neon.tech/bf?sslmode=require&channel_binding=require" in alembic[1]
    assert all(c == "runuser bfapp" for c in calls if c.startswith("runuser"))
    assert calls.index("systemctl restart bf-webapp.target") > calls.index(alembic[-1])
    smoke = [c for c in calls if c.startswith("curl ")]
    assert {c.split()[1] for c in smoke} == {"bf02.test:8443:127.0.0.1"}
    assert [c.split()[2] for c in smoke] == [
        "/api/health", "/api/version", "/api/health/deep", "/legacy/_stcore/health", "/", "/_app/version.json"
    ]  # fmt: skip
    assert "not present in this release" in done.stdout and f"commit {commit[:7]}" in done.stdout
    assert re.fullmatch(r"\S+Z update v1\.0\.0 from=- result=installed seconds=\d+", host.log()[-1])
    assert not (host.root / "lock" / "bf-webapp-release").exists()


def _posix(path: Path) -> str:
    """The path as bash prints it (cygpath -u on Windows)."""
    done = _run_bash("-c", 'cygpath -u "$1" 2>/dev/null || printf %s "$1"', "bash", path.as_posix())
    return done.stdout.strip()


def test_failed_smoke_test_rolls_back_without_a_manual_step(host: Host) -> None:
    for tag in ("v1.0.0", "v1.1.0", "v1.2.0"):
        host.publish(tag)
    assert host.run("v1.0.0", "--smoke-timeout", "5").returncode == 0
    (host.storage / "furnacemind" / "SKILLS.md").write_text("edited on the plant\n", encoding="utf-8")
    done = host.run("v1.1.0", "--smoke-timeout", "5")
    assert done.returncode == 0, _out(done)
    assert (host.link("current"), host.link("previous")) == ("v1.1.0", "v1.0.0")
    assert (host.storage / "furnacemind" / "SKILLS.md").read_text(encoding="utf-8") == "edited on the plant\n"

    # v1.2.0's twin-api answers 502: the smoke test fails and v1.1.0 serves again
    done = host.run("v1.2.0", "--smoke-timeout", "2", broken="v1.2.0")
    assert done.returncode == 3, _out(done)
    assert (host.link("current"), host.link("previous")) == ("v1.1.0", "v1.0.0")
    assert "smoke test failed: /api/health: HTTP 502" in done.stderr
    assert "rolled back to v1.1.0 automatically" in done.stdout
    calls = host.calls()
    assert calls.count("systemctl restart bf-webapp.target") == 2
    after = calls[
        calls.index("systemctl restart bf-webapp.target", calls.index("systemctl restart bf-webapp.target") + 1) :
    ]
    assert any(c.endswith(" /_app/version.json") for c in after)  # the old release passed the whole smoke test
    assert "v1.2.0" in host.releases()  # kept for diagnosis
    assert re.search(r"update v1\.2\.0 from=v1\.1\.0 result=rolled back to v1\.1\.0 seconds=\d+$", host.log()[-1])
    assert not (host.root / "lock" / "bf-webapp-release").exists()

    # rollback.sh: back to the previous release, and back again
    done = host.run("--rollback", "--smoke-timeout", "5")
    assert done.returncode == 0, _out(done)
    assert (host.link("current"), host.link("previous")) == ("v1.0.0", "v1.1.0")
    assert not any(c.startswith(("gh ", "install_python", "python ")) for c in host.calls())
    assert host.run("--rollback", "v1.1.0", "--smoke-timeout", "5").returncode == 0
    assert (host.link("current"), host.link("previous")) == ("v1.1.0", "v1.0.0")
    assert re.search(r"rollback v1\.1\.0 from=v1\.0\.0 result=rolled back seconds=\d+$", host.log()[-1])


def test_rollback_to_a_release_that_fails_restores_the_one_before(host: Host) -> None:
    host.publish("v1.0.0")
    host.publish("v1.1.0")
    assert host.run("v1.0.0", "--smoke-timeout", "5").returncode == 0
    assert host.run("v1.1.0", "--smoke-timeout", "5").returncode == 0
    done = host.run("--rollback", "--smoke-timeout", "2", broken="v1.0.0")
    assert done.returncode == 3, _out(done)
    assert (host.link("current"), host.link("previous")) == ("v1.1.0", "v1.0.0")


def test_a_failure_before_the_switch_changes_nothing(host: Host) -> None:
    host.publish("v1.0.0")
    assert host.run("v1.0.0", "--smoke-timeout", "5").returncode == 0
    cases: list[tuple[str, dict[str, bool | str], dict[str, str], str]] = [
        ("v1.1.0", {"tamper": "frontend-build.tar.gz"}, {}, "checksum mismatch"),
        ("v1.2.0", {"lock_mismatch": True}, {}, "uv.lock differs from the release's"),
        ("v1.3.0", {}, {"sync_fail": "1"}, "uv sync / import check failed"),
        ("v1.4.0", {}, {"migrate_fail": "1"}, "alembic -n twin upgrade head failed: the schema stays at twin_0009"),
    ]
    for tag, publish, fake, message in cases:
        host.publish(tag, **publish)  # type: ignore[arg-type]
        done = host.run(tag, "--smoke-timeout", "5", **fake)
        assert done.returncode == 1, (tag, _out(done))
        assert message in done.stderr, tag
        assert host.link("current") == "v1.0.0" and host.releases() == ["v1.0.0"], tag
        assert not any(c.startswith("systemctl") for c in host.calls()), tag
        assert not (host.root / "lock" / "bf-webapp-release").exists(), tag


def test_failed_first_release_has_nothing_to_roll_back_to(host: Host) -> None:
    host.publish("v1.0.0")
    done = host.run("v1.0.0", "--smoke-timeout", "1", broken="v1.0.0")
    assert done.returncode == 1 and "no release before it to roll back to" in done.stderr


def test_failed_restart_and_failed_rollback_exit_1(host: Host) -> None:
    host.publish("v1.0.0")
    host.publish("v1.1.0")
    assert host.run("v1.0.0", "--smoke-timeout", "5").returncode == 0
    done = host.run("v1.1.0", "--smoke-timeout", "5", systemctl_rc="1")
    assert done.returncode == 1 and "the rollback to v1.0.0 failed too" in done.stderr
    assert host.link("current") == "v1.0.0"


def test_offline_artefacts_without_twin_database_and_pruning(host: Host) -> None:
    (host.root / "etc" / "env").write_text("BF_SITE=bf02.test\nBF_TWIN_DB_URL=off\nDATABASE_URL=x\n", encoding="utf-8")
    for tag in ("v1.0.0", "v1.1.0", "v1.2.0"):
        host.publish(tag)
        done = host.run(tag, "--artefacts", (host.root / "gh" / tag).as_posix(), "--keep", "2", "--smoke-timeout", "5")
        assert done.returncode == 0, _out(done)
        assert not any(c.startswith(("gh ", "python ")) for c in host.calls())  # no GitHub, no migration
        assert "no twin database" in done.stdout
        assert any(c.startswith("curl bf02.test:443:127.0.0.1 ") for c in host.calls())  # default port
    assert host.releases() == ["v1.1.0", "v1.2.0"]
    assert (host.storage / "furnacemind" / "SKILLS.md").read_text(encoding="utf-8") == "skills of v1.0.0\n"
    record = json.loads((host.opt / "releases" / "v1.2.0" / "RELEASE.json").read_text(encoding="utf-8"))
    assert record["twin_revision_before"] == record["twin_revision_after"] == "none"


def test_current_tag_is_nothing_to_do_and_previous_tag_points_to_rollback(host: Host) -> None:
    host.publish("v1.0.0")
    host.publish("v1.1.0")
    assert host.run("v1.0.0", "--smoke-timeout", "5").returncode == 0
    done = host.run("v1.0.0")
    assert done.returncode == 0 and "nothing to do" in done.stdout and host.calls() == []
    assert host.run("v1.1.0", "--smoke-timeout", "5").returncode == 0
    done = host.run("v1.0.0")
    assert done.returncode == 1 and "switch to it with rollback.sh" in done.stderr


def test_a_running_release_holds_the_lock(host: Host) -> None:
    host.publish("v1.0.0")
    lock = host.root / "lock" / "bf-webapp-release"
    done = host.run("v1.0.0", locked="1")  # the pid of a live process (the shell itself)
    assert done.returncode == 1 and "another release or rollback runs" in done.stderr and lock.exists()
    assert host.link("current") is None and host.calls() == []
    (lock / "pid").write_text("999999\n", encoding="utf-8")  # stale: the process is gone
    assert host.run("v1.0.0", "--smoke-timeout", "5").returncode == 0
