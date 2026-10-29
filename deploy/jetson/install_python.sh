#!/usr/bin/env bash
# BF-02 webapp: Python 3.12 and uv on the Jetson AGX Orin (SV-6.2, #316). Run as root after provision.sh; idempotent:
# every step reads the current state and changes only what differs, so a second run reports "no changes".
#
#   sudo deploy/jetson/install_python.sh [--uv-version 0.7.13] [--python 3.12] [--release DIR] [--no-check]
#
# Steps, in this order (main):
#   platform  aarch64, JetPack 6.x (L4T R36), Ubuntu 22.04 (glibc 2.35: the platform of `make lock-check`)
#   packages  build-essential (quickjs builds from source during uv sync), curl, ca-certificates
#   uv        /usr/local/bin/uv and uvx at the lock rule's version (0.7.13, CLAUDE.md), from the GitHub release
#             tarball checked against its SHA-256 file
#   python    uv-managed CPython 3.12 (python-build-standalone, aarch64) in /opt/bf-webapp/python; the system Python
#             of JetPack is never used (UV_PYTHON_PREFERENCE=only-managed)
#   env       /etc/bf-webapp/uv.env: the UV_* variables of the units (SV-6.3), release.sh (SV-6.4) and the operator's
#             shell (`set -a; . /etc/bf-webapp/uv.env; set +a`)
#   release   --release DIR: `uv sync --frozen` in DIR as its owner (bfapp for a release directory, the operator for a
#             checkout, who keeps their own uv cache), then the import check of the audited packages
#             (docs/jetson_wheel_audit.md) and one HiGHS LP through scipy
#
# The render_* functions and the parsers are pure: tests/test_jetson_scripts.py sources this file and calls them.
set -euo pipefail

APP_USER=bfapp
OPT_DIR=/opt/bf-webapp
DATA_DIR=/var/lib/bf-webapp
ETC_DIR=/etc/bf-webapp
UV_ENV_FILE=$ETC_DIR/uv.env
UV_ENV_FILE_MODE=640
UV_BIN_DIR=/usr/local/bin
PYTHON_DIR=$OPT_DIR/python
UV_CACHE=$DATA_DIR/cache/uv
UV_RELEASES=https://github.com/astral-sh/uv/releases/download
PACKAGES=(build-essential curl ca-certificates)
# import names of the audited packages (docs/jetson_wheel_audit.md) and the two source builds of the Makefile
AUDIT_MODULES=(torch scipy numpy pandas psycopg2 duckdb chromadb pyarrow influxdb_client_3 sentence_transformers
	xgboost onnxruntime quickjs pypika)

CHANGES=()

usage() {
	cat <<'EOF'
Usage: install_python.sh [--uv-version 0.7.13] [--python 3.12] [--release DIR] [--no-check]

Python 3.12 and uv on the BF-02 Jetson (SV-6.2). Run as root after provision.sh; safe to run again.

  --uv-version V        uv release to install (default 0.7.13, the lock rule's version)
  --python X.Y          CPython minor version uv installs and the project uses (default 3.12)
  --release DIR         run `uv sync --frozen` in DIR as bfapp (DIR owned by bfapp), then the import check
  --no-check            skip the import check after the sync
  -h, --help            this text
EOF
}

note() { printf '   %s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
die() {
	printf 'install_python.sh: %s\n' "$*" >&2
	exit 1
}
usage_error() {
	printf 'install_python.sh: %s\n\n' "$*" >&2
	usage >&2
	exit 2
}
changed() {
	CHANGES+=("$*")
	note "changed: $*"
}

# ---- pure helpers (tested) ------------------------------------------------------------------------------------------

is_version() { [[ $1 =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; }

# stdin: `uv --version` ("uv 0.7.13 (62ed17b23 2025-06-12)") or `python --version` ("Python 3.12.10"); stdout: the version
version_of() { awk 'NR == 1 && NF >= 2 && $2 ~ /^[0-9]+\.[0-9]+/ { print $2; found = 1 } END { exit !found }'; }

# same_minor 3.12.10 3.12: the full version belongs to the minor series
same_minor() { [[ $1 == "$2" || $1 == "$2".* ]]; }

# stdin: `ldd --version`; stdout: the glibc version ("2.35")
glibc_version() { awk 'NR == 1 { print $NF; exit }'; }

# uv_asset MACHINE (uname -m): the release tarball of uv for this machine
uv_asset() {
	case $1 in
	aarch64 | arm64) printf 'uv-aarch64-unknown-linux-gnu.tar.gz\n' ;;
	x86_64) printf 'uv-x86_64-unknown-linux-gnu.tar.gz\n' ;;
	*) return 1 ;;
	esac
}

uv_release_url() { printf '%s/%s/%s\n' "$UV_RELEASES" "$1" "$2"; } # version asset

# checksum_ok FILE < sha256-file: the first field of the SHA-256 file equals the file's digest
checksum_ok() {
	local want have
	want=$(awk 'NR == 1 { print tolower($1) }')
	have=$(sha256sum "$1" | awk '{ print $1 }')
	[[ -n $want && $want == "$have" ]]
}

render_uv_env() { # python-dir cache-dir python-minor
	cat <<EOF
# uv on the BF-02 Jetson (deploy/jetson/install_python.sh, SV-6.2). EnvironmentFile= of the bf-* units; read by
# release.sh; in a shell: set -a; . /etc/bf-webapp/uv.env; set +a
UV_PYTHON_INSTALL_DIR=$1
UV_PYTHON=$3
UV_PYTHON_PREFERENCE=only-managed
UV_PYTHON_DOWNLOADS=never
UV_CACHE_DIR=$2
UV_LINK_MODE=copy
UV_FROZEN=1
UV_NO_PROGRESS=1
EOF
}

# uv_env_for USER < uv.env: its KEY=VALUE lines for USER; a user other than bfapp keeps their own uv cache (the shared
# one is bfapp's, mode 750)
uv_env_for() {
	local own=0
	[[ $1 == "$APP_USER" ]] && own=1
	awk -v own="$own" '/^[[:space:]]*(#|$)/ { next } !own && /^UV_CACHE_DIR=/ { next } { print }'
}

# stdout: Python that imports every audited package, prints its version, and solves one LP with HiGHS through scipy
import_check_script() {
	local modules
	modules=$(printf '"%s", ' "${AUDIT_MODULES[@]}")
	cat <<EOF
import importlib, platform, sys
modules = [${modules%, }]
failed = []
print(f"python {platform.python_version()} {platform.machine()} {sys.executable}")
for name in modules:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - every failure is reported
        failed.append(name)
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        continue
    print(f"ok   {name} {getattr(module, '__version__', '')}")
import torch
print(f"torch cuda available: {torch.cuda.is_available()} (CPU wheel from PyPI: False)")
from scipy.optimize import linprog
lp = linprog(c=[1.0, 2.0], A_ub=[[-1.0, -1.0]], b_ub=[-1.0], bounds=[(0, None), (0, None)], method="highs")
print(f"HiGHS through scipy: status {lp.status}, objective {lp.fun}")
failed += [] if lp.status == 0 and abs(lp.fun - 1.0) < 1e-9 else ["scipy-highs"]
sys.exit(1 if failed else 0)
EOF
}

# ---- system helpers -------------------------------------------------------------------------------------------------

write_file() { # path mode owner:group < content; WROTE=1 when the content changed
	local path=$1 mode=$2 owner=$3 tmp
	WROTE=0
	mkdir -p "$(dirname "$path")"
	tmp=$(mktemp "$(dirname "$path")/.$(basename "$path").XXXXXX")
	cat >"$tmp"
	if [[ -f $path ]] && cmp -s "$tmp" "$path"; then
		rm -f "$tmp"
	else
		mv -f "$tmp" "$path"
		WROTE=1
	fi
	chown "$owner" "$path"
	chmod "$mode" "$path"
}

as_user() { # user command...: the command as user, in the current directory, with the variables of uv.env
	local user=$1 home vars
	shift
	home=$(getent passwd "$user" | cut -d: -f6)
	mapfile -t vars < <(uv_env_for "$user" <"$UV_ENV_FILE")
	runuser -u "$user" -- env "HOME=${home:-/}" "${vars[@]}" "$@"
}

# the uv-managed interpreter in PYTHON_DIR, found from / without VIRTUAL_ENV (no project .venv)
managed_python() {
	(cd / && env -u VIRTUAL_ENV UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" UV_PYTHON_PREFERENCE=only-managed \
		"$UV_BIN_DIR/uv" python find "$PYTHON_MINOR")
}

# ---- steps ----------------------------------------------------------------------------------------------------------

step_platform() {
	step "platform: aarch64, JetPack 6.x, glibc"
	local ubuntu glibc
	[[ $(uname -m) == aarch64 ]] || die "machine is $(uname -m), not aarch64: run this on the Jetson"
	[[ -r /etc/nv_tegra_release ]] || die "/etc/nv_tegra_release is missing: not a JetPack (L4T) system"
	grep -q '^# R36 ' /etc/nv_tegra_release || die "not L4T R36 (JetPack 6.x): run provision.sh first"
	ubuntu=$(. /etc/os-release && printf '%s' "$VERSION_ID")
	[[ $ubuntu == 22.04 ]] || die "Ubuntu $ubuntu; JetPack 6 is Ubuntu 22.04"
	glibc=$(ldd --version 2>/dev/null | glibc_version)
	note "Ubuntu $ubuntu, glibc $glibc (wheels resolved for manylinux_2_35)"
	[[ -d $ETC_DIR && -d $DATA_DIR/cache ]] || die "$ETC_DIR or $DATA_DIR/cache is missing: run provision.sh first"
	id "$APP_USER" >/dev/null 2>&1 || die "user $APP_USER is missing: run provision.sh first"
}

step_packages() {
	step "packages: ${PACKAGES[*]}"
	local pkg missing=()
	for pkg in "${PACKAGES[@]}"; do
		[[ $(dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null) == "install ok installed" ]] || missing+=("$pkg")
	done
	if ((${#missing[@]} == 0)); then
		note "installed"
		return
	fi
	DEBIAN_FRONTEND=noninteractive apt-get update -q ||
		die "apt-get update failed: the Jetson needs the Ubuntu/NVIDIA mirrors (or a plant mirror) for ${missing[*]}"
	DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends "${missing[@]}"
	changed "installed ${missing[*]}"
}

step_uv() {
	step "uv $UV_VERSION in $UV_BIN_DIR"
	local have="" asset url tmp
	[[ -x $UV_BIN_DIR/uv ]] && have=$("$UV_BIN_DIR/uv" --version 2>/dev/null | version_of || true)
	if [[ $have == "$UV_VERSION" && -x $UV_BIN_DIR/uvx ]]; then
		note "uv $have installed"
		return
	fi
	asset=$(uv_asset "$(uname -m)") || die "no uv release asset for $(uname -m)"
	url=$(uv_release_url "$UV_VERSION" "$asset")
	tmp=$(mktemp -d)
	trap 'rm -rf -- "${tmp:-}"; trap - RETURN' RETURN # a RETURN trap outlives the function: it clears itself
	curl -fsSL --retry 3 -o "$tmp/$asset" "$url" || die "download failed: $url (GitHub reachable? a plant mirror?)"
	curl -fsSL --retry 3 -o "$tmp/$asset.sha256" "$url.sha256" || die "download failed: $url.sha256"
	checksum_ok "$tmp/$asset" <"$tmp/$asset.sha256" || die "SHA-256 of $asset does not match $url.sha256"
	tar -xzf "$tmp/$asset" -C "$tmp" --strip-components=1
	install -m 755 -o root -g root "$tmp/uv" "$UV_BIN_DIR/uv"
	install -m 755 -o root -g root "$tmp/uvx" "$UV_BIN_DIR/uvx"
	have=$("$UV_BIN_DIR/uv" --version | version_of) || die "$UV_BIN_DIR/uv does not run"
	[[ $have == "$UV_VERSION" ]] || die "installed uv reports $have, not $UV_VERSION"
	changed "uv $have ($asset, SHA-256 checked)"
}

step_python() {
	step "python $PYTHON_MINOR (uv-managed) in $PYTHON_DIR"
	local have=""
	mkdir -p "$PYTHON_DIR"
	have=$(managed_python 2>/dev/null || true)
	if [[ -n $have ]]; then
		note "$("$have" --version 2>/dev/null) at $have"
	else
		UV_PYTHON_INSTALL_DIR=$PYTHON_DIR "$UV_BIN_DIR/uv" python install "$PYTHON_MINOR" ||
			die "uv python install $PYTHON_MINOR failed (python-build-standalone from GitHub; UV_PYTHON_INSTALL_MIRROR for a plant mirror)"
		have=$(managed_python) || die "python $PYTHON_MINOR not found after the install"
		changed "$("$have" --version) at $have"
	fi
	same_minor "$("$have" --version | version_of)" "$PYTHON_MINOR" || die "$have is not Python $PYTHON_MINOR"
	# bfapp runs the interpreter; root owns it
	chown -R root:root "$PYTHON_DIR"
	chmod 755 "$PYTHON_DIR"
	[[ -z $(find "$PYTHON_DIR" -not -perm -o=r -print -quit) ]] || chmod -R o+rX "$PYTHON_DIR"
}

step_env() {
	step "env: $UV_ENV_FILE"
	mkdir -p "$UV_CACHE"
	chown "$APP_USER:$APP_USER" "$UV_CACHE"
	chmod 750 "$UV_CACHE"
	render_uv_env "$PYTHON_DIR" "$UV_CACHE" "$PYTHON_MINOR" | write_file "$UV_ENV_FILE" "$UV_ENV_FILE_MODE" "root:$APP_USER"
	if ((WROTE)); then changed "$UV_ENV_FILE"; else note "unchanged"; fi
}

step_release() {
	step "release: uv sync --frozen in $RELEASE_DIR"
	local owner
	[[ -f $RELEASE_DIR/uv.lock && -f $RELEASE_DIR/pyproject.toml ]] || die "$RELEASE_DIR holds no uv.lock / pyproject.toml"
	owner=$(stat -c %U "$RELEASE_DIR")
	[[ $owner != root ]] || die "$RELEASE_DIR is owned by root: the sync runs as the owner (bfapp, or you for a checkout)"
	note "as $owner$([[ $owner == "$APP_USER" ]] && printf ' (uv cache %s)' "$UV_CACHE" || printf ' (own uv cache)')"
	(cd "$RELEASE_DIR" && as_user "$owner" "$UV_BIN_DIR/uv" sync --frozen) || die "uv sync --frozen failed in $RELEASE_DIR"
	note "synced: $(cd "$RELEASE_DIR" && as_user "$owner" "$UV_BIN_DIR/uv" run --no-sync python --version)"
	if ((CHECK)); then
		(cd "$RELEASE_DIR" && import_check_script | as_user "$owner" "$UV_BIN_DIR/uv" run --no-sync python -) ||
			die "import check failed: see the FAIL lines (docs/jetson_wheel_audit.md)"
		note "import check passed (${#AUDIT_MODULES[@]} packages, HiGHS LP)"
	fi
}

summary() {
	step "summary"
	if ((${#CHANGES[@]})); then note "${#CHANGES[@]} change(s)"; else note "no changes: the system already matched"; fi
	note "next: deploy/jetson/benchmark.sh --open-issue (runtime benchmark, docs/jetson_benchmark.md)"
}

need_value() { [[ $# -ge 2 && $2 != -* ]] || usage_error "$1 needs a value"; }

parse_args() {
	UV_VERSION=0.7.13
	PYTHON_MINOR=3.12
	RELEASE_DIR=""
	CHECK=1
	while (($#)); do
		case $1 in
		--uv-version)
			need_value "$@"
			UV_VERSION=$2
			shift 2
			;;
		--python)
			need_value "$@"
			PYTHON_MINOR=$2
			shift 2
			;;
		--release)
			need_value "$@"
			RELEASE_DIR=${2%/}
			shift 2
			;;
		--no-check)
			CHECK=0
			shift
			;;
		-h | --help)
			usage
			exit 0
			;;
		*) usage_error "unknown argument: $1" ;;
		esac
	done
	is_version "$UV_VERSION" || usage_error "--uv-version: not a version (X.Y.Z): $UV_VERSION"
	[[ $PYTHON_MINOR =~ ^3\.[0-9]+$ ]] || usage_error "--python: a minor version (3.12): $PYTHON_MINOR"
	[[ -z $RELEASE_DIR || -d $RELEASE_DIR ]] || usage_error "--release: not a directory: $RELEASE_DIR"
}

main() {
	parse_args "$@"
	[[ $EUID -eq 0 ]] || die "run as root (sudo)"
	step_platform
	step_packages
	step_uv
	step_python
	step_env
	[[ -z $RELEASE_DIR ]] || step_release
	summary
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
