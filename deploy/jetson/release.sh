#!/usr/bin/env bash
# BF-02 webapp: install a release on the Jetson, or roll back to an installed one (SV-6.4, #318). Run as root after
# provision.sh and install_python.sh. One update or rollback at a time (lock directory /run/lock/bf-webapp-release).
#
#   sudo deploy/jetson/release.sh <tag> [--repo OWNER/NAME] [--artefacts DIR] [--keep N] [--smoke-timeout S] [--no-check]
#   sudo deploy/jetson/rollback.sh [<tag>]      (= release.sh --rollback [<tag>]; default: the previous release)
#
# Steps of an update, in this order (main):
#   fetch     the CI release of <tag> (.github/workflows/release.yml): frontend-build.tar.gz, uv.lock, pnpm-lock.yaml,
#             SHA256SUMS, and the tag's source archive (gh release download); --artefacts DIR: the same five files
#             fetched elsewhere (source archive as source.tar.gz)
#   verify    sha256sum --check SHA256SUMS, which must list the three files; the source archive's uv.lock and
#             frontend/pnpm-lock.yaml equal the release's (the source is the commit CI built)
#   unpack    the source into releases/<tag> (staged, then renamed), the SPA build into releases/<tag>/frontend/build;
#             src/storage -> /var/lib/bf-webapp/storage (seeded with the release's files it does not hold yet);
#             owner bfapp (uv sync runs as bfapp)
#   sync      the release's install_python.sh --release: uv sync --frozen as bfapp, then the import check
#   migrate   alembic -n twin upgrade head as bfapp with /etc/bf-webapp/env (skipped without a twin database)
#   record    releases/<tag>/RELEASE.json: tag, commit (from the source archive), SPA build version, installation time,
#             twin schema revision before and after; the twin-api serves it at /api/version (footer of the SPA)
#   switch    current -> releases/<tag> (a new symlink renamed over the old one), previous -> the replaced release
#   restart   systemctl restart bf-webapp.target
#   smoke     through Caddy (https://BF_SITE:BF_HTTPS_PORT at 127.0.0.1), retried until --smoke-timeout: /api/health;
#             /api/version names <tag> and its commit (the new processes answer); /api/health/deep when present;
#             /legacy/_stcore/health; the SPA index; _app/version.json = the release's build
#   rollback  automatic when the restart or the smoke test fails: current -> the release before, restart, smoke test
#   prune     the newest --keep release directories stay (current and previous always)
#
# A failure before the switch changes nothing that runs (exit 1; the new directory is removed). The twin schema is
# never downgraded: a rollback runs the older code on the newer schema (migrations add, they do not drop); RELEASE.json
# keeps the revision before the upgrade for a manual `alembic -n twin downgrade <revision>`.
# Exit status: 0 installed (or rolled back by rollback.sh), 1 failed (nothing switched, or the rollback failed too),
# 2 usage, 3 the update failed and the release before it serves again.
#
# The pure helpers are tested, and the whole update and rollback run against stubbed gh, systemctl, curl and runuser
# with a failed smoke test (tests/test_jetson_release.py sources this file, sets the paths and calls main).
set -euo pipefail

APP_USER=bfapp
OPT_DIR=/opt/bf-webapp
DATA_DIR=/var/lib/bf-webapp
ETC_DIR=/etc/bf-webapp
UNIT_DIR=/etc/systemd/system
LOCK_DIR=/run/lock/bf-webapp-release
SERVICES_TARGET=bf-webapp.target
DEFAULT_REPO=saikumar-voptimai/evonith_bf_webapp
ASSETS=(frontend-build.tar.gz uv.lock pnpm-lock.yaml) # listed in SHA256SUMS (release.yml)
SOURCE_ARCHIVE=source.tar.gz
SMOKE_RETRY_S=3
CURL_TIMEOUT_S=5
TARGET_S=300 # update and rollback each below 5 min (#318)
PATH_ENV=/usr/local/bin:/usr/bin:/bin

T0=$SECONDS
SWITCHED=0
NEW_DIR=""    # created by this run; removed when the run ends before the switch
WORK_DIR=""   # downloads
REASON=""     # why the last restart or smoke test failed
OK_DETAIL=""  # set by a smoke check that passes with a remark
LAST=""       # last failure of a smoke check
SMOKE_DEADLINE=0

usage() {
	cat <<'EOF'
Usage: release.sh <tag> [--repo OWNER/NAME] [--artefacts DIR] [--keep N] [--smoke-timeout S] [--no-check]
       release.sh --rollback [<tag>] [--smoke-timeout S]      (rollback.sh [<tag>])

Install a CI release of the BF-02 webapp on the Jetson, or roll back (SV-6.4). Run as root.

  <tag>                 release tag, e.g. v1.2.0 (the GitHub release made by release.yml)
  --repo OWNER/NAME     repository of the release (default saikumar-voptimai/evonith_bf_webapp; gh needs a login or
                        GH_TOKEN: sudo --preserve-env=GH_TOKEN ...)
  --artefacts DIR       take the files from DIR instead of GitHub: frontend-build.tar.gz, uv.lock, pnpm-lock.yaml,
                        SHA256SUMS and the tag's source archive as source.tar.gz
  --keep N              release directories to keep, newest first (default 3, at least 2); current and previous stay
  --smoke-timeout S     seconds the smoke test may take after the restart (default 180)
  --no-check            skip the import check after uv sync (install_python.sh --no-check)
  --rollback            switch to an installed release: <tag>, else the previous one; restart and smoke test
  -h, --help            this text

Exit status: 0 done, 1 failed (nothing switched, or the rollback failed too), 2 usage, 3 the update failed and the
release before it serves again (automatic rollback).
EOF
}

note() { printf '   %s\n' "$*"; }
step() { printf '\n== [%4ds] %s\n' "$((SECONDS - T0))" "$*"; }
warn() { printf '   WARN %s\n' "$*" >&2; }
die() {
	printf 'release.sh: %s\n' "$*" >&2
	exit 1
}
usage_error() {
	printf 'release.sh: %s\n\n' "$*" >&2
	usage >&2
	exit 2
}

# ---- pure helpers (tested) ------------------------------------------------------------------------------------------

# a release tag: v + version, at most 80 characters, safe as a directory name and in JSON (services/release.py)
valid_tag() { [[ $1 =~ ^v[0-9][0-9A-Za-z._+-]{0,78}$ ]]; }

valid_repo() { [[ $1 =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; }

# sums_cover NAME... < SHA256SUMS: each name has one line "<64 hex>  <name>"
sums_cover() {
	local sums name
	sums=$(cat)
	for name in "$@"; do
		[[ $(grep -cE "^[0-9a-fA-F]{64} [ *]${name//./\\.}\$" <<<"$sums") -eq 1 ]] || return 1
	done
}

# stdin: a tar.gz from `git archive` (GitHub's source archives); stdout: the commit of its pax global header
tar_commit_id() {
	{ gzip -dc 2>/dev/null || true; } | head -c 1024 | LC_ALL=C grep -ao 'comment=[0-9a-f]\{40\}' | head -n 1 |
		cut -d= -f2 | grep .
}

# json_string KEY < JSON: the string value of KEY (flat JSON: RELEASE.json, version.json, /api/version)
json_string() {
	LC_ALL=C sed -n 's/.*"'"$1"'"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1 | grep .
}

# stdin: an EnvironmentFile (KEY=VALUE, comments, blank lines, values in matching quotes); stdout: KEY=VALUE lines
# for env(1). Never sourced: a value such as a database URL with & or ; is passed as it is.
env_assignments() {
	local line key value
	while IFS= read -r line || [[ -n $line ]]; do
		line=${line%$'\r'}
		[[ $line =~ ^[[:space:]]*(#|$) ]] && continue
		[[ $line =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
		key=${BASH_REMATCH[2]}
		value=${BASH_REMATCH[3]}
		if [[ $value =~ ^\"(.*)\"$ || $value =~ ^\'(.*)\'$ ]]; then value=${BASH_REMATCH[1]}; fi
		printf '%s=%s\n' "$key" "$value"
	done
}

# env_value KEY < assignments: the last value of KEY (empty when absent)
env_value() { awk -v k="$1" 'index($0, k "=") == 1 { v = substr($0, length(k) + 2) } END { print v }'; }

# stdin: assignments; exit 0 when the twin store has a database (data/twin/engine.py: BF_TWIN_DB_URL, else
# DATABASE_URL; BF_TWIN_DB_URL off / none / 0 / false disables it)
twin_db_configured() {
	local vars twin
	vars=$(cat)
	twin=$(env_value BF_TWIN_DB_URL <<<"$vars")
	case ${twin,,} in off | none | 0 | false) return 1 ;; esac
	[[ -n $twin || -n $(env_value DATABASE_URL <<<"$vars") ]]
}

# stdin: output of `alembic current` (log lines and "<revision> (head)"); stdout: the revision; exit 1 at base
alembic_revision() {
	awk '$1 ~ /^(INFO|WARNING|WARN|DEBUG|ERROR)/ { next } $1 ~ /^[A-Za-z0-9_]+$/ { print $1; found = 1; exit }
		END { exit !found }'
}

# prune_list KEEP CURRENT PREVIOUS < names, newest first: the names to remove (dot names are staging, skipped)
prune_list() {
	awk -v keep="$1" -v cur="$2" -v prev="$3" '$0 == "" || /^\./ { next }
		{ n++; if (n > keep && $0 != cur && $0 != prev) print }'
}

render_release_json() { # tag commit build installed-at revision-before revision-after
	cat <<EOF
{
  "tag": "$1",
  "commit": "$2",
  "frontend_build": "$3",
  "installed_at": "$4",
  "twin_revision_before": "$5",
  "twin_revision_after": "$6",
  "installed_by": "deploy/jetson/release.sh"
}
EOF
}

# a value from a release record or build file that is safe to compare and to write into JSON
safe_name() { [[ $1 =~ ^[0-9A-Za-z][0-9A-Za-z._+-]{0,79}$ ]]; }

duration() { printf '%d min %02d s' "$(($1 / 60))" "$(($1 % 60))"; }

# ---- system helpers -------------------------------------------------------------------------------------------------

require_root() { [[ $EUID -eq 0 ]] || die "run as root (sudo)"; }

release_dir() { printf '%s/releases/%s\n' "$OPT_DIR" "$1"; }

link_tag() { # link: the release a symlink of OPT_DIR names (empty when absent)
	local target
	[[ -L $OPT_DIR/$1 ]] || return 0
	target=$(readlink "$OPT_DIR/$1")
	printf '%s\n' "${target##*/}"
}

current_tag() { link_tag current; }
previous_tag() { link_tag previous; }

acquire_lock() {
	local pid
	mkdir -p "$(dirname "$LOCK_DIR")"
	if ! mkdir "$LOCK_DIR" 2>/dev/null; then
		pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
		if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
			die "another release or rollback runs (pid $pid, $LOCK_DIR)"
		fi
		rm -rf "$LOCK_DIR"
		mkdir "$LOCK_DIR" || die "cannot take the lock $LOCK_DIR"
	fi
	printf '%s\n' "$$" >"$LOCK_DIR/pid"
	trap cleanup EXIT
}

cleanup() {
	local status=$?
	if [[ -n $WORK_DIR ]]; then rm -rf "$WORK_DIR"; fi
	rm -rf "$OPT_DIR/releases/.stage-"* 2>/dev/null || true
	if ((!SWITCHED)) && [[ -n $NEW_DIR && -d $NEW_DIR ]]; then
		rm -rf "$NEW_DIR"
		printf '   removed %s (not switched to)\n' "$NEW_DIR" >&2
	fi
	rm -rf "$LOCK_DIR"
	exit "$status"
}

log_result() { # action tag from result
	printf '%s %s %s from=%s result=%s seconds=%d\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:--}" "$4" \
		"$((SECONDS - T0))" >>"$OPT_DIR/release.log"
}

link_atomic() { # target link: a new symlink renamed over the link (readers see the old or the new one)
	local tmp="$2.new.$$"
	rm -f "$tmp"
	ln -s "$1" "$tmp"
	mv -Tf "$tmp" "$2"
}

set_links() { # current-tag [previous-tag]
	link_atomic "releases/$1" "$OPT_DIR/current"
	if [[ -n ${2:-} ]]; then
		link_atomic "releases/$2" "$OPT_DIR/previous"
	else
		rm -f "$OPT_DIR/previous"
	fi
}

# as_app DIR ASSIGNMENT... -- COMMAND...: COMMAND as bfapp in DIR with only these variables (and PATH, HOME)
as_app() {
	local dir=$1 vars=()
	shift
	while (($#)) && [[ $1 != -- ]]; do
		vars+=("$1")
		shift
	done
	shift
	(cd "$dir" && runuser -u "$APP_USER" -- env -i "PATH=$PATH_ENV" "HOME=$DATA_DIR" "${vars[@]}" "$@")
}

app_env() { # the assignments of the units: /etc/bf-webapp/env and uv.env
	env_assignments <"$ETC_DIR/env"
	if [[ -f $ETC_DIR/uv.env ]]; then env_assignments <"$ETC_DIR/uv.env"; fi
}

# http_get PATH: HTTP_CODE and HTTP_BODY of https://BF_SITE:BF_HTTPS_PORT/PATH through Caddy at 127.0.0.1
http_get() {
	local out
	out=$(curl -sk -m "$CURL_TIMEOUT_S" --resolve "$SITE:$PORT:127.0.0.1" -w '\n%{http_code}' \
		"https://$SITE:$PORT$1" 2>/dev/null) || true
	HTTP_CODE=${out##*$'\n'}
	HTTP_BODY=${out%$'\n'*}
}

# ---- smoke test -----------------------------------------------------------------------------------------------------

check_health() {
	http_get /api/health
	[[ $HTTP_CODE == 200 && $(json_string status <<<"$HTTP_BODY" || true) == ok ]] ||
		{ LAST="/api/health: HTTP ${HTTP_CODE:-none}"; return 1; }
}

check_version() { # tag commit: /api/version names them (the processes of this release answer)
	local release sha
	http_get /api/version
	[[ $HTTP_CODE == 200 ]] || { LAST="/api/version: HTTP ${HTTP_CODE:-none}"; return 1; }
	release=$(json_string release <<<"$HTTP_BODY" || true)
	sha=$(json_string git_sha <<<"$HTTP_BODY" || true)
	[[ $release == "$1" ]] || { LAST="/api/version names release '${release:-null}', not $1"; return 1; }
	[[ $2 == unknown || $sha == "$2" ]] || { LAST="/api/version names commit $sha, not $2"; return 1; }
	OK_DETAIL="release $release, commit ${sha:0:7}"
}

check_deep() { # /api/health/deep when the twin-api has it
	http_get /api/health/deep
	case $HTTP_CODE in
	2??) OK_DETAIL="HTTP $HTTP_CODE" ;;
	404) OK_DETAIL="not present in this release" ;;
	401 | 403) OK_DETAIL="needs a session (HTTP $HTTP_CODE): not checked" ;;
	*)
		LAST="/api/health/deep: HTTP ${HTTP_CODE:-none}"
		return 1
		;;
	esac
}

check_legacy() {
	http_get /legacy/_stcore/health
	[[ $HTTP_CODE == 200 ]] || { LAST="/legacy/_stcore/health: HTTP ${HTTP_CODE:-none}"; return 1; }
}

check_index() {
	http_get /
	[[ $HTTP_CODE == 200 && $HTTP_BODY == *"<html"* ]] || { LAST="SPA index: HTTP ${HTTP_CODE:-none}"; return 1; }
}

check_build() { # build: Caddy serves this release's SPA build
	local have
	http_get /_app/version.json
	have=$(json_string version <<<"$HTTP_BODY" || true)
	[[ $HTTP_CODE == 200 && $have == "$1" ]] ||
		{ LAST="_app/version.json: HTTP ${HTTP_CODE:-none}, build '${have}', not $1"; return 1; }
	OK_DETAIL="build $have"
}

smoke_check() { # label check [args]: retried every SMOKE_RETRY_S until the deadline
	local label=$1
	shift
	OK_DETAIL=""
	until "$@"; do
		if ((SECONDS >= SMOKE_DEADLINE)); then
			REASON="smoke test failed: $LAST"
			note "FAIL $label: $LAST"
			return 1
		fi
		sleep "$SMOKE_RETRY_S"
		OK_DETAIL=""
	done
	note "ok   $label${OK_DETAIL:+ ($OK_DETAIL)}"
}

smoke_test() { # tag
	local dir tag=$1 commit=unknown build=""
	dir=$(release_dir "$tag")
	if [[ -f $dir/RELEASE.json ]]; then
		commit=$(json_string commit <"$dir/RELEASE.json" || printf 'unknown')
	fi
	build=$(json_string version <"$dir/frontend/build/_app/version.json" 2>/dev/null || true)
	SMOKE_DEADLINE=$((SECONDS + SMOKE_TIMEOUT_S))
	note "https://$SITE:$PORT at 127.0.0.1, up to ${SMOKE_TIMEOUT_S} s"
	smoke_check "twin-api /api/health" check_health || return 1
	if [[ -f $dir/RELEASE.json ]]; then
		smoke_check "twin-api /api/version" check_version "$tag" "$commit" || return 1
	else
		note "skip twin-api /api/version: $dir has no RELEASE.json"
	fi
	smoke_check "twin-api /api/health/deep" check_deep || return 1
	smoke_check "Streamlit /legacy/_stcore/health" check_legacy || return 1
	smoke_check "SPA index" check_index || return 1
	if [[ -n $build ]]; then
		smoke_check "SPA _app/version.json" check_build "$build" || return 1
	fi
}

restart_and_smoke() { # tag: restart the services on the current link, then the smoke test
	step "restart: systemctl restart $SERVICES_TARGET ($1)"
	if ! systemctl restart "$SERVICES_TARGET"; then
		REASON="systemctl restart $SERVICES_TARGET failed"
		note "FAIL $REASON"
		return 1
	fi
	step "smoke test: $1"
	smoke_test "$1"
}

# activate NEW OLD OLD_PREVIOUS: current -> NEW (previous -> OLD), restart, smoke test; on a failure back to OLD
# (previous -> OLD_PREVIOUS). Returns 0 NEW serves, 3 OLD serves again, 1 neither passed the smoke test.
activate() {
	local new=$1 old=$2 old_previous=$3
	step "switch: current -> releases/$new${old:+ (previous -> releases/$old)}"
	trap '' INT TERM HUP # from here the run ends with one release serving: no interruption
	set_links "$new" "$old"
	SWITCHED=1
	restart_and_smoke "$new" && return 0
	local failure=$REASON
	if [[ -z $old ]]; then
		REASON="$failure; no release before it to roll back to"
		return 1
	fi
	step "rollback: current -> releases/$old ($new: $failure)"
	set_links "$old" "$old_previous"
	if restart_and_smoke "$old"; then
		REASON=$failure
		return 3
	fi
	REASON="$failure; the rollback to $old failed too: $REASON"
	return 1
}

journal_hint() {
	note "logs: journalctl -u bf-twin-api -u bf-twin-worker -u bf-streamlit -u bf-caddy --since -15min"
}

# ---- update steps ---------------------------------------------------------------------------------------------------

step_fetch() {
	step "fetch: $TAG${ARTEFACTS_DIR:+ from $ARTEFACTS_DIR}${ARTEFACTS_DIR:- from github.com/$REPO}"
	local name
	WORK_DIR=$(mktemp -d "$OPT_DIR/releases/.stage-fetch.XXXXXX")
	if [[ -n $ARTEFACTS_DIR ]]; then
		for name in "${ASSETS[@]}" SHA256SUMS "$SOURCE_ARCHIVE"; do
			[[ -f $ARTEFACTS_DIR/$name ]] || die "$ARTEFACTS_DIR/$name is missing"
			cp "$ARTEFACTS_DIR/$name" "$WORK_DIR/$name"
		done
	else
		command -v gh >/dev/null 2>&1 ||
			die "gh is not installed: install it (and gh auth login, or GH_TOKEN), or fetch the files elsewhere and pass --artefacts DIR"
		gh release download "$TAG" --repo "$REPO" --dir "$WORK_DIR" --clobber \
			--pattern frontend-build.tar.gz --pattern uv.lock --pattern pnpm-lock.yaml --pattern SHA256SUMS ||
			die "gh release download $TAG --repo $REPO failed (release made by release.yml? gh logged in?)"
		gh release download "$TAG" --repo "$REPO" --archive tar.gz --output "$WORK_DIR/$SOURCE_ARCHIVE" --clobber ||
			die "the source archive of $TAG could not be downloaded"
	fi
	for name in "${ASSETS[@]}" SHA256SUMS "$SOURCE_ARCHIVE"; do
		note "$name $(wc -c <"$WORK_DIR/$name" | tr -d ' ') bytes"
	done
}

step_verify() {
	step "verify: SHA256SUMS"
	sums_cover "${ASSETS[@]}" <"$WORK_DIR/SHA256SUMS" || die "SHA256SUMS does not list ${ASSETS[*]} once each"
	(cd "$WORK_DIR" && sha256sum --check --strict --quiet SHA256SUMS) || die "checksum mismatch: see above"
	note "${ASSETS[*]}: SHA-256 match"
	COMMIT=$(tar_commit_id <"$WORK_DIR/$SOURCE_ARCHIVE" || printf 'unknown')
	note "source archive of commit $COMMIT"
}

step_unpack() {
	step "unpack: $(release_dir "$TAG")"
	local stage dir build
	stage="$OPT_DIR/releases/.stage-$TAG"
	dir=$(release_dir "$TAG")
	rm -rf "$stage"
	mkdir -p "$stage/frontend"
	tar -xzf "$WORK_DIR/$SOURCE_ARCHIVE" -C "$stage" --strip-components=1 --no-same-owner ||
		die "cannot unpack the source archive"
	[[ -f $stage/pyproject.toml && -f $stage/src/api/main.py && -x $stage/deploy/jetson/install_python.sh ]] ||
		die "the source archive is not the BF-02 webapp (pyproject.toml, src/api/main.py, deploy/jetson/install_python.sh)"
	cmp -s "$stage/uv.lock" "$WORK_DIR/uv.lock" || die "the source's uv.lock differs from the release's: not the commit CI built"
	cmp -s "$stage/frontend/pnpm-lock.yaml" "$WORK_DIR/pnpm-lock.yaml" ||
		die "the source's frontend/pnpm-lock.yaml differs from the release's: not the commit CI built"
	note "source matches the release's lock files"
	rm -rf "$stage/frontend/build"
	tar -xzf "$WORK_DIR/frontend-build.tar.gz" -C "$stage/frontend" --no-same-owner || die "cannot unpack the SPA build"
	[[ -f $stage/frontend/build/200.html && -f $stage/frontend/build/_app/version.json ]] ||
		die "the SPA build has no 200.html or _app/version.json"
	build=$(json_string version <"$stage/frontend/build/_app/version.json" || true)
	safe_name "$build" || die "the SPA build's _app/version.json names no version"
	BUILD=$build
	[[ $BUILD == "$TAG" ]] || note "SPA build $BUILD (built before release.yml named builds by their tag)"
	if [[ -e $dir ]]; then # an earlier attempt of this tag, not the current release (checked in main)
		rm -rf "$dir"
		note "replaced the earlier directory of $TAG"
	fi
	mv "$stage" "$dir"
	NEW_DIR=$dir
	touch "$dir"
	link_storage "$dir"
	chown -R "$APP_USER:$APP_USER" "$dir"
	chmod 755 "$dir"
	note "source, SPA build $BUILD, owner $APP_USER"
}

link_storage() { # release-dir: src/storage -> DATA_DIR/storage, seeded with the release's files the state lacks
	local store="$DATA_DIR/storage" src="$1/src/storage" file seeded=0
	mkdir -p "$store"
	if [[ -d $src && ! -L $src ]]; then
		while IFS= read -r -d '' file; do
			file=${file#./}
			[[ -e $store/$file ]] && continue
			mkdir -p "$(dirname "$store/$file")"
			cp -p "$src/$file" "$store/$file"
			seeded=$((seeded + 1))
		done < <(cd "$src" && find . -type f -print0)
		rm -rf "$src"
	fi
	ln -sfn "$store" "$src"
	chown -R "$APP_USER:$APP_USER" "$store"
	note "src/storage -> $store ($seeded file(s) seeded from the release; the state is kept)"
}

step_sync() {
	local dir args=()
	dir=$(release_dir "$TAG")
	((CHECK)) || args+=(--no-check)
	step "sync: install_python.sh --release $dir ${args[*]}"
	"$dir/deploy/jetson/install_python.sh" --release "$dir" "${args[@]}" || die "uv sync / import check failed (above)"
}

step_migrate() {
	local dir vars out
	dir=$(release_dir "$TAG")
	step "migrate: alembic -n twin upgrade head"
	mapfile -t vars < <(app_env)
	REV_BEFORE=none
	REV_AFTER=none
	if ! printf '%s\n' "${vars[@]}" | twin_db_configured; then
		note "no twin database (BF_TWIN_DB_URL off, or neither it nor DATABASE_URL in $ETC_DIR/env): skipped"
		return
	fi
	out=$(as_app "$dir" "${vars[@]}" -- ./.venv/bin/python -m alembic -n twin current 2>&1) ||
		die "alembic -n twin current failed (database reachable?): $(tail -n 3 <<<"$out")"
	REV_BEFORE=$(alembic_revision <<<"$out" || printf 'base')
	as_app "$dir" "${vars[@]}" -- ./.venv/bin/python -m alembic -n twin upgrade head ||
		die "alembic -n twin upgrade head failed: the schema stays at $REV_BEFORE (one transaction)"
	out=$(as_app "$dir" "${vars[@]}" -- ./.venv/bin/python -m alembic -n twin current 2>&1) || true
	REV_AFTER=$(alembic_revision <<<"$out" || printf 'unknown')
	note "twin schema $REV_BEFORE -> $REV_AFTER"
}

step_record() {
	local dir
	dir=$(release_dir "$TAG")
	step "record: $dir/RELEASE.json"
	render_release_json "$TAG" "$COMMIT" "$BUILD" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$REV_BEFORE" "$REV_AFTER" \
		>"$dir/RELEASE.json"
	chown "$APP_USER:$APP_USER" "$dir/RELEASE.json"
	note "$TAG, commit $COMMIT, SPA build $BUILD"
}

units_note() { # release-dir: unit files of the release that differ from the installed ones
	local unit differ=()
	for unit in "$1"/deploy/jetson/systemd/*; do
		[[ -f $unit ]] || continue
		cmp -s "$unit" "$UNIT_DIR/${unit##*/}" || differ+=("${unit##*/}")
	done
	if ((${#differ[@]})); then
		warn "units of this release differ from $UNIT_DIR: ${differ[*]}; install them with this release's provision.sh"
	fi
}

step_prune() {
	step "prune: keep $KEEP release(s)"
	local name removed=0
	while IFS= read -r name; do
		rm -rf "$(release_dir "$name")"
		note "removed releases/$name"
		removed=$((removed + 1))
	done < <(ls -1t "$OPT_DIR/releases" | prune_list "$KEEP" "$(current_tag)" "$(previous_tag)")
	((removed)) || note "nothing to remove"
}

finish() { # action tag from result status
	local took=$((SECONDS - T0))
	log_result "$1" "$2" "$3" "$4"
	step "summary"
	note "$1 $2: $4 in $(duration "$took") (target below $(duration "$TARGET_S"))"
	((took < TARGET_S)) || warn "took $(duration "$took"), above the $(duration "$TARGET_S") target"
	local prev
	prev=$(previous_tag)
	note "current -> releases/$(current_tag), previous -> ${prev:+releases/}${prev:-none}"
}

# ---- entry points ---------------------------------------------------------------------------------------------------

need_value() { [[ $# -ge 2 && $2 != -* ]] || usage_error "$1 needs a value"; }

parse_args() {
	TAG=""
	ROLLBACK=0
	REPO=$DEFAULT_REPO
	ARTEFACTS_DIR=""
	KEEP=3
	SMOKE_TIMEOUT_S=180
	CHECK=1
	local update_only=()
	while (($#)); do
		case $1 in
		--repo)
			need_value "$@"
			REPO=$2
			update_only+=("$1")
			shift 2
			;;
		--artefacts)
			need_value "$@"
			ARTEFACTS_DIR=${2%/}
			update_only+=("$1")
			shift 2
			;;
		--keep)
			need_value "$@"
			KEEP=$2
			update_only+=("$1")
			shift 2
			;;
		--smoke-timeout)
			need_value "$@"
			SMOKE_TIMEOUT_S=$2
			shift 2
			;;
		--no-check)
			CHECK=0
			update_only+=("$1")
			shift
			;;
		--rollback)
			ROLLBACK=1
			shift
			;;
		-h | --help)
			usage
			exit 0
			;;
		-*) usage_error "unknown argument: $1" ;;
		*)
			[[ -z $TAG ]] || usage_error "one tag only: $TAG $1"
			TAG=$1
			shift
			;;
		esac
	done
	if ((ROLLBACK)); then
		((${#update_only[@]} == 0)) || usage_error "--rollback takes a tag and --smoke-timeout only, not ${update_only[*]}"
	else
		[[ -n $TAG ]] || usage_error "the release tag is required"
	fi
	[[ -z $TAG ]] || valid_tag "$TAG" || usage_error "not a release tag (v1.2.0): $TAG"
	valid_repo "$REPO" || usage_error "--repo: OWNER/NAME: $REPO"
	[[ -z $ARTEFACTS_DIR || -d $ARTEFACTS_DIR ]] || usage_error "--artefacts: not a directory: $ARTEFACTS_DIR"
	[[ $KEEP =~ ^[0-9]+$ ]] && ((KEEP >= 2)) || usage_error "--keep: a whole number of at least 2: $KEEP"
	[[ $SMOKE_TIMEOUT_S =~ ^[0-9]+$ ]] && ((SMOKE_TIMEOUT_S >= 1 && SMOKE_TIMEOUT_S <= 1800)) ||
		usage_error "--smoke-timeout: seconds, 1 to 1800: $SMOKE_TIMEOUT_S"
}

load_site() { # BF_SITE and BF_HTTPS_PORT of the units (the Caddyfile's defaults otherwise)
	local vars
	[[ -f $ETC_DIR/env ]] || die "$ETC_DIR/env is missing: run provision.sh first"
	vars=$(app_env)
	SITE=$(env_value BF_SITE <<<"$vars")
	PORT=$(env_value BF_HTTPS_PORT <<<"$vars")
	SITE=${SITE:-bf02.local}
	PORT=${PORT:-443}
}

main_update() {
	local old old_previous status
	old=$(current_tag)
	old_previous=$(previous_tag)
	if [[ $old == "$TAG" ]]; then
		note "$TAG is the current release: nothing to do (rollback.sh switches to another one)"
		return 0
	fi
	if [[ $old_previous == "$TAG" && -d $(release_dir "$TAG") ]]; then
		die "$TAG is the previous release: switch to it with rollback.sh (a reinstall would replace the rollback target)"
	fi
	note "update to $TAG; current ${old:-none}, previous ${old_previous:-none}"
	step_fetch
	step_verify
	step_unpack
	step_sync
	step_migrate
	step_record
	status=0
	activate "$TAG" "$old" "$old_previous" || status=$?
	case $status in
	0)
		units_note "$(release_dir "$TAG")"
		step_prune
		finish update "$TAG" "$old" installed
		;;
	3)
		warn "update to $TAG failed: $REASON"
		note "rolled back to $old automatically; releases/$TAG stays for diagnosis"
		journal_hint
		finish update "$TAG" "$old" "rolled back to $old"
		exit 3
		;;
	*)
		warn "update to $TAG failed: $REASON"
		journal_hint
		finish update "$TAG" "$old" failed
		exit 1
		;;
	esac
}

main_rollback() {
	local old old_previous target status dir
	old=$(current_tag)
	old_previous=$(previous_tag)
	target=${TAG:-$old_previous}
	[[ -n $target ]] || die "no previous release to roll back to; name an installed one: rollback.sh <tag>"
	[[ $target != "$old" ]] || die "$target is the current release"
	dir=$(release_dir "$target")
	[[ -x $dir/.venv/bin/python ]] || die "$dir is not an installed release (no .venv)"
	note "rollback to $target; current ${old:-none}"
	status=0
	activate "$target" "$old" "$old_previous" || status=$?
	case $status in
	0) finish rollback "$target" "$old" "rolled back" ;;
	3)
		warn "rollback to $target failed: $REASON"
		note "$old serves again"
		journal_hint
		finish rollback "$target" "$old" "failed, $old serves again"
		exit 3
		;;
	*)
		warn "rollback to $target failed: $REASON"
		journal_hint
		finish rollback "$target" "$old" failed
		exit 1
		;;
	esac
}

main() {
	parse_args "$@"
	require_root
	[[ -d $OPT_DIR/releases ]] || die "$OPT_DIR/releases is missing: run provision.sh first"
	load_site
	acquire_lock
	if ((ROLLBACK)); then main_rollback; else main_update; fi
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
