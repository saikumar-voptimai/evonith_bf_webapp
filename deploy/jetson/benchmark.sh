#!/usr/bin/env bash
# BF-02 webapp: runtime benchmark of the optimiser on the Jetson AGX Orin (SV-6.2, #316). Run as bfapp or the
# operator in a synced checkout or release directory (install_python.sh --release DIR); changes nothing on the system.
#
#   deploy/jetson/benchmark.sh [--release DIR] [--solvers 8] [--mode M ...] [--doc docs/jetson_benchmark.md]
#                              [--log-dir DIR] [--live-day] [--open-issue] [--repo owner/name]
#
# Runs scripts/optimiser_bench.py in the modes solve, jacobian (--solvers 8), step2, max_production and step1 on the
# reference plan day (COKE_ENERGY_TODAY=2026-10-15, COKE_ENERGY_WEATHER_LIVE=0, so the plans compare across hosts and
# days; --live-day uses today's), records the wall times in the results block of docs/jetson_benchmark.md (between the
# markers) and compares each with its target of section 10 of the plan (#262): one network solve 2.8 s, the Jacobian
# 6 s on 8 solver processes, Step 2 90 s, maximum production 300 s, Step 1 600 s. A runtime above its target opens a
# GitHub issue (--open-issue, gh signed in), else the script prints the gh command. Exit status 1 when a mode is above
# its target or gave no result.
#
# The parsers and renderers are pure: tests/test_jetson_scripts.py sources this file and calls them.
set -uo pipefail

MODES_ALL=(solve jacobian step2 max_production step1)
MARK_START='<!-- benchmark:start -->'
MARK_END='<!-- benchmark:end -->'
UV_ENV_FILE=/etc/bf-webapp/uv.env
REFERENCE_DAY=2026-10-15
ISSUE_REF='#316'
US=$'\x1f' # field separator of the records of runtimes above target

usage() {
	cat <<'EOF'
Usage: benchmark.sh [--release DIR] [--solvers N] [--mode M ...] [--doc FILE] [--log-dir DIR] [--live-day]
                    [--open-issue] [--repo OWNER/NAME]

Optimiser runtime benchmark of the BF-02 Jetson (SV-6.2): scripts/optimiser_bench.py per mode, the results block of
docs/jetson_benchmark.md, one GitHub issue per runtime above its target.

  --release DIR         checkout or release directory with the synced .venv (default: this script's repository)
  --solvers N           solver processes of the Jacobian mode (default 8, the target's pool); the other modes use
                        BF_STRATA_SOLVERS, else min(8, cpus)
  --mode M              one of solve, jacobian, step2, max_production, step1 (repeat; default: all five, in this order)
  --doc FILE            the benchmark document to update (default docs/jetson_benchmark.md of the release)
  --log-dir DIR         where the bench output of every mode is kept (default /tmp/bf-benchmark-<timestamp>)
  --live-day            plan day and blast moisture of today (default: the reference day 2026-10-15, shipped moisture)
  --open-issue          open a GitHub issue (gh) for each runtime above its target; default: print the command
  --repo OWNER/NAME     repository of the issues (default: the release's origin remote)
  -h, --help            this text
EOF
}

note() { printf '   %s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
die() {
	printf 'benchmark.sh: %s\n' "$*" >&2
	exit 1
}
usage_error() {
	printf 'benchmark.sh: %s\n\n' "$*" >&2
	usage >&2
	exit 2
}

# ---- pure helpers (tested) ------------------------------------------------------------------------------------------

# target_seconds MODE: section 10 of the plan (#262), AGX Orin column
target_seconds() {
	case $1 in
	solve) printf '2.8\n' ;;
	jacobian) printf '6\n' ;;
	step2) printf '90\n' ;;
	max_production) printf '300\n' ;;
	step1) printf '600\n' ;;
	*) return 1 ;;
	esac
}

mode_label() {
	case $1 in
	solve) printf 'One STRATA-Flow network solve (worst of 3 repeats after the first)\n' ;;
	jacobian) printf 'Jacobian at a new calibration (17 solves) on the solver pool\n' ;;
	step2) printf 'Step 2, STRATA-ω and Fixed-ω in parallel, with the value of limits\n' ;;
	max_production) printf 'Maximum production (STRATA-ω)\n' ;;
	step1) printf 'Step 1, cheapest production level and frontier (STRATA-ω)\n' ;;
	*) return 1 ;;
	esac
}

# bench_args MODE SOLVERS: arguments of scripts/optimiser_bench.py; the limit is the mode's target
bench_args() {
	local target
	target=$(target_seconds "$1") || return 1
	case $1 in
	jacobian) printf -- '--mode %s --solvers %s --limit-s %s\n' "$1" "$2" "$target" ;;
	*) printf -- '--mode %s --limit-s %s\n' "$1" "$target" ;;
	esac
}

# stdin: the bench output; stdout: the measured seconds of the mode (empty, status 1, when the line is missing)
bench_seconds() {
	case $1 in
	solve) sed -nE 's/^one network solve: .* then [0-9.]+-([0-9.]+) s.*$/\1/p' ;;
	jacobian) sed -nE 's/^network Jacobian .* then the Jacobian ([0-9.]+) s.*$/\1/p' ;;
	*) sed -nE 's/^\[parallel\] wall time of the solves ([0-9.]+) s.*$/\1/p' ;;
	esac | awk 'NR == 1 { print; found = 1 } END { exit !found }'
}

# stdin: the bench output; stdout: one line of context for the notes column
bench_note() {
	case $1 in
	solve) sed -nE 's/^one network solve: ([0-9.]+) s the first .*, then ([0-9.-]+) s.*$/first solve \1 s (builds the context), then \2 s/p' ;;
	jacobian) sed -nE 's/^network Jacobian \(17 solves\): serial ([0-9.]+) s; ([0-9]+) solver processes started in ([0-9.]+) s.*$/serial \1 s; \2 processes started in \3 s/p' ;;
	*) awk '/^\[parallel\] .* network solves/ { n += $6; s++ } END { if (s) printf "%d network solves in %d case(s)\n", n, s }' ;;
	esac | head -n 1
}

# above_target SECONDS TARGET
above_target() { awk -v s="$1" -v t="$2" 'BEGIN { exit !(s + 0 > t + 0) }'; }

fmt_seconds() { # seconds -> "48.0 s" or "1158 s (19.3 min)"
	awk -v s="$1" 'BEGIN { if (s + 0 >= 120) printf "%.0f s (%.1f min)\n", s, s / 60; else printf "%.1f s\n", s }'
}

# stdin: results TSV (mode, seconds or "-", target, status, note); stdout: the markdown rows
results_rows() {
	local mode seconds target status note label measured
	while IFS=$'\t' read -r mode seconds target status note; do
		label=$(mode_label "$mode")
		if [[ $seconds == - ]]; then measured="—"; else measured=$(fmt_seconds "$seconds"); fi
		printf '| %s | %s | %s | %s | %s |\n' "$label" "$(fmt_seconds "$target")" "$measured" "$status" "${note:-—}"
	done
}

# results_block HOST-LINE < results TSV: the whole block between the markers
results_block() {
	printf '%s\n' "$MARK_START" "Measured $1" "" \
		"| Operation | Target (AGX Orin) | Measured | Status | Notes |" "|---|---|---|---|---|"
	results_rows
	printf '%s\n' "$MARK_END"
}

# replace_between FILE < new block: FILE with the lines from the start marker to the end marker replaced by stdin
replace_between() {
	grep -qF "$MARK_START" "$1" && grep -qF "$MARK_END" "$1" || return 1
	awk -v start="$MARK_START" -v end="$MARK_END" -v block="$(cat)" '
		index($0, start) == 1 { print block; skip = 1; next }
		index($0, end) == 1 { skip = 0; next }
		!skip { print }' "$1"
}

# repo_of_url URL: owner/name of a GitHub remote (ssh or https)
repo_of_url() { sed -nE 's#^(git@github\.com:|https?://github\.com/)([^/]+/[^/]+)/?$#\2#p' <<<"$1" | sed 's/\.git$//'; }

issue_title() { printf 'Jetson benchmark: %s %s above the %s target (SV-6.2)\n' "$1" "$(fmt_seconds "$2")" "$(fmt_seconds "$3")"; } # mode seconds target

issue_body() { # mode seconds target host-line note log-file
	cat <<EOF
\`deploy/jetson/benchmark.sh\` measured **$(mode_label "$1")** at **$(fmt_seconds "$2")**, above the section-10 target of
$(fmt_seconds "$3") (plan #262, AGX Orin column).

Measured $4
Bench: ${5:-—}
Log: \`$6\`

Next: profile the mode on the device (\`scripts/optimiser_bench.py --mode $1 --compare\`), then either reduce the solves
of the mode or move its target with the owner.

Refs $ISSUE_REF
EOF
}

# ---- run ------------------------------------------------------------------------------------------------------------

host_line() {
	local model cpus power uv py commit
	model=$(tr -d '\0' 2>/dev/null </proc/device-tree/model || uname -m)
	cpus=$(nproc 2>/dev/null || printf '?')
	power=$( { nvpmodel -q 2>/dev/null || true; } | sed -n 's/^NV Power Mode: //p' | head -n 1)
	uv=$(uv --version 2>/dev/null | awk '{ print $2 }')
	py=$(uv run --no-sync python --version 2>/dev/null | awk '{ print $2 }')
	commit=$(git describe --always --dirty 2>/dev/null || printf '?')
	printf '%s on %s, %s CPUs%s, uv %s, Python %s, commit %s, solver processes %s' "$(date -u +'%Y-%m-%d %H:%MZ')" \
		"$model" "$cpus" "${power:+, nvpmodel $power}" "${uv:-?}" "${py:-?}" "$commit" "${BF_STRATA_SOLVERS:-min(8, cpus)}"
}

run_mode() { # mode -> appends to RESULTS
	local mode=$1 log="$LOG_DIR/$mode.log" args seconds note status rc target
	target=$(target_seconds "$mode")
	read -ra args <<<"$(bench_args "$mode" "$SOLVERS")"
	step "$mode: optimiser_bench.py ${args[*]}"
	PYTHONPATH=src uv run --no-sync python -u scripts/optimiser_bench.py "${args[@]}" 2>&1 | tee "$log"
	rc=${PIPESTATUS[0]}
	seconds=$(bench_seconds "$mode" <"$log") || seconds=""
	note=$(bench_note "$mode" <"$log")
	if [[ -z $seconds ]]; then
		status="FAIL no result (exit $rc)"
		seconds=-
		FAILS=$((FAILS + 1))
	elif above_target "$seconds" "$target"; then
		status="FAIL above target"
		FAILS=$((FAILS + 1))
		ABOVE+=("$mode$US$seconds$US$target$US$note$US$log") # not tabs: read would merge an empty note's tabs
	else
		status="PASS within target"
	fi
	printf '%s\t%s\t%s\t%s\t%s\n' "$mode" "$seconds" "$target" "$status" "$note" >>"$RESULTS"
	note "$status: $([[ $seconds == - ]] && printf 'no time' || printf '%s s' "$seconds") (target $target s)"
}

open_issues() {
	local line mode seconds target note log title body_file
	((${#ABOVE[@]})) || return 0
	step "issues: ${#ABOVE[@]} runtime(s) above target"
	for line in "${ABOVE[@]}"; do
		IFS=$US read -r mode seconds target note log <<<"$line"
		title=$(issue_title "$mode" "$seconds" "$target")
		body_file="$LOG_DIR/issue-$mode.md"
		issue_body "$mode" "$seconds" "$target" "$HOST" "$note" "$log" >"$body_file"
		if ((OPEN_ISSUE)); then
			gh issue create --repo "$REPO" --title "$title" --body-file "$body_file" || die "gh issue create failed ($title)"
		else
			note "gh issue create --repo $REPO --title \"$title\" --body-file $body_file"
		fi
	done
}

need_value() { [[ $# -ge 2 && $2 != -* ]] || usage_error "$1 needs a value"; }

parse_args() {
	RELEASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
	SOLVERS=8
	MODES=()
	DOC=""
	LOG_DIR=""
	LIVE_DAY=0
	OPEN_ISSUE=0
	REPO=""
	local mode
	while (($#)); do
		case $1 in
		--release)
			need_value "$@"
			RELEASE_DIR=${2%/}
			shift 2
			;;
		--solvers)
			need_value "$@"
			SOLVERS=$2
			shift 2
			;;
		--mode)
			need_value "$@"
			MODES+=("$2")
			shift 2
			;;
		--doc)
			need_value "$@"
			DOC=$2
			shift 2
			;;
		--log-dir)
			need_value "$@"
			LOG_DIR=$2
			shift 2
			;;
		--live-day)
			LIVE_DAY=1
			shift
			;;
		--open-issue)
			OPEN_ISSUE=1
			shift
			;;
		--repo)
			need_value "$@"
			REPO=$2
			shift 2
			;;
		-h | --help)
			usage
			exit 0
			;;
		*) usage_error "unknown argument: $1" ;;
		esac
	done
	[[ $SOLVERS =~ ^[0-9]+$ ]] || usage_error "--solvers: a number of processes: $SOLVERS"
	((${#MODES[@]})) || MODES=("${MODES_ALL[@]}")
	for mode in "${MODES[@]}"; do
		target_seconds "$mode" >/dev/null || usage_error "--mode: one of ${MODES_ALL[*]}: $mode"
	done
	[[ -d $RELEASE_DIR ]] || usage_error "--release: not a directory: $RELEASE_DIR"
	[[ -z $DOC ]] && DOC=docs/jetson_benchmark.md
	[[ -z $LOG_DIR ]] && LOG_DIR=${TMPDIR:-/tmp}/bf-benchmark-$(date -u +%Y%m%d-%H%M%S)
}

main() {
	parse_args "$@"
	cd "$RELEASE_DIR" || die "cannot enter $RELEASE_DIR"
	[[ -f scripts/optimiser_bench.py && -f uv.lock ]] || die "$RELEASE_DIR is not a checkout (scripts/optimiser_bench.py)"
	if [[ -r $UV_ENV_FILE ]]; then
		set -a
		# shellcheck disable=SC1090
		. "$UV_ENV_FILE"
		set +a
	fi
	if [[ -n ${UV_CACHE_DIR:-} && ! -w $UV_CACHE_DIR ]]; then unset UV_CACHE_DIR; fi # bfapp's cache; others use their own
	command -v uv >/dev/null || die "uv is not installed (deploy/jetson/install_python.sh)"
	[[ -x .venv/bin/python ]] || die "no .venv in $RELEASE_DIR: install_python.sh --release $RELEASE_DIR"
	if ((LIVE_DAY)); then
		unset COKE_ENERGY_TODAY
		export COKE_ENERGY_WEATHER_LIVE=1
	else
		export COKE_ENERGY_TODAY=$REFERENCE_DAY COKE_ENERGY_WEATHER_LIVE=0
	fi
	if [[ -z $REPO ]]; then
		REPO=$(repo_of_url "$(git remote get-url origin 2>/dev/null)")
		[[ -n $REPO ]] || REPO=saikumar-voptimai/evonith_bf_webapp
	fi
	if ((OPEN_ISSUE)); then
		command -v gh >/dev/null && gh auth status >/dev/null 2>&1 || die "--open-issue needs gh signed in (gh auth login)"
	fi
	mkdir -p "$LOG_DIR" || die "cannot create $LOG_DIR"
	RESULTS=$LOG_DIR/results.tsv
	: >"$RESULTS"
	FAILS=0
	ABOVE=()
	HOST=$(host_line)
	step "benchmark: $HOST"
	note "plan day ${COKE_ENERGY_TODAY:-today} (COKE_ENERGY_WEATHER_LIVE=$COKE_ENERGY_WEATHER_LIVE), modes ${MODES[*]}, logs $LOG_DIR"
	local mode
	for mode in "${MODES[@]}"; do run_mode "$mode"; done

	step "results: $DOC"
	if [[ ! -f $DOC ]]; then
		note "$DOC is missing; results in $RESULTS"
	elif [[ ! -w $DOC || ! -w $(dirname "$DOC") ]]; then
		note "$DOC is not writable by $(id -un); results in $RESULTS"
	elif ((${#MODES[@]} < ${#MODES_ALL[@]})); then
		note "partial run (${MODES[*]}): the block of $DOC is left as it is; results in $RESULTS"
	elif results_block "$HOST" <"$RESULTS" | replace_between "$DOC" >"$DOC.tmp"; then
		mv -f "$DOC.tmp" "$DOC"
		note "results block written (commit $DOC)"
	else
		rm -f "$DOC.tmp"
		note "markers $MARK_START / $MARK_END missing in $DOC; results in $RESULTS"
		FAILS=$((FAILS + 1))
	fi
	results_block "$HOST" <"$RESULTS"
	open_issues
	((FAILS == 0)) && note "every mode within its target" || note "$FAILS mode(s) above target or without result"
	return $((FAILS > 0))
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
