#!/usr/bin/env bash
# BF-02 webapp: health check of the bf-* services (SV-6.3, #317). Installed by provision.sh as
# /usr/local/sbin/bf-healthcheck and run as root by bf-healthcheck.timer every 10 s (bf-healthcheck.service).
#
# A process that exits is restarted by systemd itself (Restart=always, 2 s). This check catches one that runs but no
# longer answers: every service of PROBES that is active and older than its start grace is probed, and a service that
# fails FAIL_LIMIT probes in a row is restarted (systemctl restart). A service that is not active (stopped by hand,
# starting, waiting for its restart) is left alone.
#
#   unit                     grace  probe
#   bf-redis.service          15 s  redis-cli PING on BF_REDIS_URL (redis://127.0.0.1:6379/0)
#   bf-twin-api.service       90 s  GET http://127.0.0.1:8100/api/health
#   bf-streamlit.service      90 s  GET http://127.0.0.1:8501/legacy/_stcore/health
#   bf-furnace-data.service   60 s  GET http://127.0.0.1:8000/health
#   bf-caddy.service          15 s  HTTPS on 127.0.0.1:BF_HTTPS_PORT (443) with the name BF_SITE (bf02.local): any
#                                   HTTP answer (the upstreams have their own probes)
#   bf-twin-worker.service   120 s  heartbeat twin:worker:alive at most 60 s old; not judged while Redis does not answer
#
# Output only for failed probes and restarts, at warning level (bf-healthcheck.service logs nothing below warning).
# The pure functions are tested: tests/test_jetson_units.py sources this file and calls them.
set -uo pipefail

STATE_DIR=/run/bf-healthcheck # RuntimeDirectory= of bf-healthcheck.service: counters survive between runs, not a boot
UPTIME_FILE=/proc/uptime
FAIL_LIMIT=2
PROBE_TIMEOUT_S=3
WORKER_HEARTBEAT_KEY=twin:worker:alive
WORKER_HEARTBEAT_MAX_S=60
# unit grace-seconds probe [argument]; the units of deploy/jetson/systemd that run a server (tested)
PROBES=(
	"bf-redis.service 15 redis"
	"bf-twin-api.service 90 http http://127.0.0.1:8100/api/health"
	"bf-streamlit.service 90 http http://127.0.0.1:8501/legacy/_stcore/health"
	"bf-furnace-data.service 60 http http://127.0.0.1:8000/health"
	"bf-caddy.service 15 https"
	"bf-twin-worker.service 120 heartbeat"
)

warn() { printf '<4>%s\n' "$*"; } # syslog level prefix: warning

# ---- pure helpers (tested) ------------------------------------------------------------------------------------------

# decide STATE AGE_S GRACE_S PROBE FAILS LIMIT: "<action> <failures>"; PROBE is ok, fail or skip (not judged).
# Actions: idle (not active, in its grace, or not judged), ok, fail (count it), restart (LIMIT failures in a row).
decide() {
	local state=$1 age=$2 grace=$3 probe=$4 fails=$5 limit=$6
	if [[ $state != active || $probe == skip ]] || ((age < grace)); then
		printf 'idle 0\n'
	elif [[ $probe == ok ]]; then
		printf 'ok 0\n'
	elif ((fails + 1 >= limit)); then
		printf 'restart 0\n'
	else
		printf 'fail %d\n' $((fails + 1))
	fi
}

# unit_age_s ACTIVE_ENTER_MONOTONIC_US UPTIME_S: whole seconds since the unit became active (0 when it never did)
unit_age_s() {
	awk -v m="$1" -v u="$2" 'BEGIN { if (m + 0 <= 0) { print 0; exit } a = u - m / 1e6; printf "%d\n", (a < 0 ? 0 : a) }'
}

# heartbeat_age_s ISO_TIME NOW_EPOCH_S: seconds since the heartbeat (the worker writes datetime.now(UTC).isoformat())
heartbeat_age_s() {
	local t
	t=$(date -d "$1" +%s 2>/dev/null) || return 1
	printf '%d\n' $(($2 - t))
}

# ---- probes ---------------------------------------------------------------------------------------------------------

redis_url() { printf '%s\n' "${BF_REDIS_URL:-redis://127.0.0.1:6379/0}"; }

probe_redis() { [[ $(timeout "$PROBE_TIMEOUT_S" redis-cli -u "$(redis_url)" ping 2>/dev/null) == PONG ]]; }

probe_http() { curl -fsS -o /dev/null -m "$PROBE_TIMEOUT_S" "$1" 2>/dev/null; }

probe_https() {
	local site=${BF_SITE:-bf02.local} port=${BF_HTTPS_PORT:-443} code
	code=$(curl -sk -o /dev/null -w '%{http_code}' -m "$PROBE_TIMEOUT_S" --resolve "$site:$port:127.0.0.1" \
		"https://$site:$port/" 2>/dev/null)
	[[ $code =~ ^[1-5][0-9][0-9]$ ]]
}

# exit 0 fresh, 1 stale or missing, 2 not judged (Redis does not answer)
probe_heartbeat() {
	local raw age
	probe_redis || return 2
	raw=$(timeout "$PROBE_TIMEOUT_S" redis-cli -u "$(redis_url)" --raw get "$WORKER_HEARTBEAT_KEY" 2>/dev/null)
	[[ -n $raw ]] || return 1
	age=$(heartbeat_age_s "$raw" "$(date +%s)") || return 1
	((age <= WORKER_HEARTBEAT_MAX_S))
}

run_probe() { # kind [argument]: prints ok, fail or skip
	local status
	"probe_$1" "${@:2}"
	status=$?
	case $status in
	0) printf 'ok\n' ;;
	2) printf 'skip\n' ;;
	*) printf 'fail\n' ;;
	esac
}

main() {
	local spec unit grace kind arg state mono uptime age probe fails action
	mkdir -p "$STATE_DIR"
	read -r uptime _ <"$UPTIME_FILE"
	for spec in "${PROBES[@]}"; do
		read -r unit grace kind arg <<<"$spec"
		state=$(systemctl is-active "$unit" 2>/dev/null)
		mono=$(systemctl show -p ActiveEnterTimestampMonotonic --value "$unit" 2>/dev/null)
		age=$(unit_age_s "${mono:-0}" "$uptime")
		probe=skip
		if [[ $state == active ]] && ((age >= grace)); then
			probe=$(run_probe "$kind" ${arg:+"$arg"})
		fi
		fails=$(cat "$STATE_DIR/$unit" 2>/dev/null || printf '0')
		[[ $fails =~ ^[0-9]+$ ]] || fails=0
		read -r action fails < <(decide "$state" "$age" "$grace" "$probe" "$fails" "$FAIL_LIMIT")
		printf '%s\n' "$fails" >"$STATE_DIR/$unit"
		case $action in
		fail) warn "$unit: $kind probe failed ($fails of $FAIL_LIMIT in a row)" ;;
		restart)
			warn "$unit: $FAIL_LIMIT $kind probes failed in a row, restarting it"
			systemctl restart --no-block "$unit" || warn "$unit: systemctl restart failed"
			;;
		esac
	done
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
