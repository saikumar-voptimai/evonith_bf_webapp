#!/usr/bin/env bash
# BF-02 webapp: checks of the Jetson base system (SV-6.1, #315). Run as root on the Orin after provision.sh, and again
# after a reboot. Changes nothing except its record of the last passing boot (/var/lib/bf-webapp/verify/last_pass).
#
#   sudo deploy/jetson/verify_base.sh [--max-offset-ms 100] [--admin-subnet CIDR ...] [--no-http] [--unit NAME ...]
#
# One line per check: PASS, WARN, FAIL or INFO. Exit status 1 when a check fails, else 0.
#   platform  L4T R36 (JetPack 6.x)
#   clock     chronyc tracking: synchronised and |system time offset| < --max-offset-ms (100 ms)
#   nvme      /var/lib/bf-webapp mounted from an NVMe, by UUID in /etc/fstab
#   power     nvpmodel in the MAXN mode; bf-jetson-power.service enabled and active
#   layout    user bfapp; directories and /etc/bf-webapp/env with their owner and mode; current -> releases/<tag>
#   logs      logrotate (14 days) accepted by logrotate --debug; journald retention 14 days
#   firewall  ufw active, incoming denied; open: 443/tcp (+ 80/tcp) from anywhere, 22/tcp from admin subnets only
#   ports     listening sockets, reachable from the network or loopback only (the open-port list)
#   units     base units, bf-* units, caddy and redis-server (when installed) enabled and active; system running
#   reboot    PASS when every check passes on a boot other than the last passing one: services back after a reboot
#
# The parsers are pure: tests/test_jetson_scripts.py sources this file and calls them.
set -uo pipefail

APP_USER=bfapp
OPT_DIR=/opt/bf-webapp
DATA_DIR=/var/lib/bf-webapp
ENV_FILE=/etc/bf-webapp/env
ENV_FILE_MODE=640
POWER_UNIT=bf-jetson-power.service
LOGROTATE_CONF=/etc/logrotate.d/bf-webapp
JOURNALD_CONF=/etc/systemd/journald.conf.d/bf-webapp.conf
STATE_FILE=$DATA_DIR/verify/last_pass
BASE_UNITS=(ssh.service chrony.service ufw.service "$POWER_UNIT" logrotate.timer)
# tcp/udp ports a process may listen on beyond loopback; ufw still decides what is reachable
EXPOSED_OK="tcp/22 tcp/80 tcp/443 udp/68 udp/546 udp/5353"
# path owner:group mode; provision.sh holds the same table (tested)
LAYOUT_DIRS=(
	"/opt/bf-webapp root:root 755"
	"/opt/bf-webapp/releases root:root 755"
	"/var/lib/bf-webapp bfapp:bfapp 750"
	"/var/lib/bf-webapp/storage bfapp:bfapp 750"
	"/var/lib/bf-webapp/cache bfapp:bfapp 750"
	"/var/log/bf-webapp bfapp:bfapp 750"
	"/etc/bf-webapp root:bfapp 750"
)

PASSES=0
WARNS=0
FAILS=0

usage() {
	cat <<'EOF'
Usage: verify_base.sh [--max-offset-ms MS] [--admin-subnet CIDR ...] [--no-http] [--unit NAME ...]

Checks the base system of the BF-02 Jetson (SV-6.1). Run as root; run again after a reboot.

  --max-offset-ms MS    largest accepted clock offset from NTP time (default 100)
  --admin-subnet CIDR   the SSH rules must name exactly these networks (default: any network, never anywhere)
  --no-http             80/tcp must be closed (provision.sh --no-http)
  --unit NAME           another unit that must be enabled and active (repeat for more)
  -h, --help            this text
EOF
}

result() { # level check text
	printf '%-4s  %-8s  %s\n' "$1" "$2" "$3"
	case $1 in
	PASS) PASSES=$((PASSES + 1)) ;;
	WARN) WARNS=$((WARNS + 1)) ;;
	FAIL) FAILS=$((FAILS + 1)) ;;
	esac
}

# ---- pure parsers (tested) ------------------------------------------------------------------------------------------

# stdin: `chronyc tracking`; $1: field name, e.g. "Leap status"; stdout: its value
tracking_value() { awk -v k="$1" 'BEGIN { FS = " *: " } $1 == k { print $2; exit }'; }

# stdin: `chronyc tracking`; stdout: |system time offset| in ms, 3 decimals
tracking_offset_ms() {
	awk 'BEGIN { FS = " *: " } $1 == "System time" { split($2, a, " "); v = a[1] * 1000; if (v < 0) v = -v
		printf "%.3f\n", v; found = 1; exit } END { if (!found) exit 1 }'
}

# stdin: `chronyc tracking`; stdout: upper bound of the clock error in ms (|offset| + root dispersion + root delay / 2)
tracking_error_bound_ms() {
	awk 'BEGIN { FS = " *: " } { split($2, a, " ") } $1 == "System time" { o = a[1] < 0 ? -a[1] : a[1] }
		$1 == "Root delay" { d = a[1] } $1 == "Root dispersion" { r = a[1] }
		END { printf "%.3f\n", (o + r + d / 2) * 1000 }'
}

# stdin: `chronyc tracking`; status 0 when chronyd follows an NTP source
tracking_synced() {
	local text ref leap
	text=$(cat)
	ref=$(tracking_value "Reference ID" <<<"$text")
	leap=$(tracking_value "Leap status" <<<"$text")
	ref=${ref%% *}
	[[ -n $ref && $ref != 00000000 && $ref != 7F7F0101 && -n $leap && $leap != "Not synchronised" ]]
}

# stdin: `ufw status`; stdout: "<to><TAB><action><TAB><from>" per rule, "(v6)" dropped (an IPv4 + IPv6 pair is one rule)
ufw_rules_table() {
	awk 'BEGIN { FS = "  +" } /^--/ { body = 1; next }
		body && NF >= 3 { to = $1; from = $3; sub(/ \(v6\)$/, "", to); sub(/ \(v6\)$/, "", from)
			row = to "\t" $2 "\t" from; if (!(row in seen)) { seen[row] = 1; print row } }'
}

# stdin: ufw_rules_table; args: http(0|1) admin-subnets...; stdout: one line per problem, nothing when the rules match
firewall_problems() {
	local http=$1 to action from open443=0 open80=0 ssh=0 cidr known
	shift
	while IFS=$'\t' read -r to action from; do
		[[ $action == ALLOW* || $action == LIMIT* ]] || continue
		case $to in
		443/tcp) [[ $from != Anywhere ]] || open443=1 ;;
		80/tcp)
			if ((http)); then open80=1; else printf '80/tcp is open from %s (--no-http)\n' "$from"; fi
			;;
		22/tcp)
			if [[ $from == Anywhere ]]; then
				printf '22/tcp (SSH) is open from anywhere\n'
			elif (($#)); then
				known=0
				for cidr in "$@"; do [[ $from != "$cidr" ]] || known=1; done
				if ((known)); then ssh=1; else printf '22/tcp is open from %s, not an admin subnet\n' "$from"; fi
			else
				ssh=1
			fi
			;;
		*) printf 'unexpected rule: %s %s from %s\n' "$to" "$action" "$from" ;;
		esac
	done
	((open443)) || printf '443/tcp is not open\n'
	((!http || open80)) || printf '80/tcp (HTTPS redirect) is not open\n'
	((ssh)) || printf 'no SSH rule for an admin subnet\n'
}

# stdin: `ss -H -tulnp`; stdout: "<proto><TAB><port><TAB>loopback|exposed<TAB><address><TAB><process>"
listeners_table() {
	local netid state recvq sendq laddr peer rest addr port scope proc re='\(\("([^"]+)"'
	while read -r netid state recvq sendq laddr peer rest; do
		[[ -n ${laddr:-} ]] || continue
		port=${laddr##*:}
		addr=${laddr%:*}
		addr=${addr#[}
		addr=${addr%]}
		addr=${addr%%\%*}
		proc=unknown
		if [[ ${rest:-} =~ $re ]]; then proc=${BASH_REMATCH[1]}; fi
		case $addr in
		127.* | ::1 | ::ffff:127.*) scope=loopback ;;
		*) scope=exposed ;;
		esac
		printf '%s\t%s\t%s\t%s\t%s\n' "$netid" "$port" "$scope" "$addr" "$proc"
	done
}

# stdin: listeners_table; $1: scope; stdout: "proto/port process" once per port, sorted by port
listeners_summary() {
	awk -F '\t' -v s="$1" '$3 == s { k = $1 "/" $2; if (!(k in seen)) { seen[k] = 1; print $2 "\t" k " " $5 } }' |
		sort -n -k1,1 | cut -f2
}

# stdin: listeners_table; stdout: exposed "proto/port process" outside EXPOSED_OK
unexpected_exposed() {
	local proto port scope addr proc
	while IFS=$'\t' read -r proto port scope addr proc; do
		[[ $scope == exposed && " $EXPOSED_OK " != *" $proto/$port "* ]] && printf '%s/%s %s\n' "$proto" "$port" "$proc"
	done | sort -u
	return 0
}

# ---- checks ---------------------------------------------------------------------------------------------------------

l4t_major() { sed -n 's/^# R\([0-9]*\) (release).*/\1/p' /etc/nv_tegra_release 2>/dev/null; }

check_platform() {
	local major model
	major=$(l4t_major)
	model=$(tr -d '\0' 2>/dev/null </proc/device-tree/model || printf 'unknown model')
	if [[ $major == 36 ]]; then
		result PASS platform "$model, $(head -n1 /etc/nv_tegra_release | cut -d, -f1,2) (JetPack 6.x)"
	else
		result FAIL platform "L4T R${major:-?} is not JetPack 6.x (L4T R36.x)"
	fi
}

check_clock() {
	local text offset bound source
	if ! text=$(chronyc tracking 2>&1); then
		result FAIL clock "chronyc tracking failed: $text"
		return
	fi
	offset=$(tracking_offset_ms <<<"$text")
	bound=$(tracking_error_bound_ms <<<"$text")
	source=$(chronyc -n sources 2>/dev/null | awk '/^\^\*/ { print $2 }')
	if [[ -z $offset ]]; then
		result FAIL clock "no system time offset in chronyc tracking"
	elif ! tracking_synced <<<"$text"; then
		result FAIL clock "not synchronised to an NTP source (reference $(tracking_value "Reference ID" <<<"$text"))"
	elif awk -v o="$offset" -v m="$MAX_OFFSET_MS" 'BEGIN { exit !(o < m) }'; then
		result PASS clock "offset $offset ms < $MAX_OFFSET_MS ms (error bound $bound ms), source ${source:-?}"
	else
		result FAIL clock "offset $offset ms >= $MAX_OFFSET_MS ms (error bound $bound ms), source ${source:-?}"
	fi
}

check_nvme() {
	local source uuid
	if ! mountpoint -q "$DATA_DIR"; then
		result FAIL nvme "$DATA_DIR is not mounted"
		return
	fi
	source=$(findmnt -n -o SOURCE --mountpoint "$DATA_DIR")
	uuid=$(findmnt -n -o UUID --mountpoint "$DATA_DIR")
	if [[ $source != /dev/nvme* ]]; then
		result FAIL nvme "$DATA_DIR is mounted from $source, not from an NVMe"
	elif ! awk -v u="UUID=$uuid" -v mp="$DATA_DIR" '$1 == u && $2 == mp { f = 1 } END { exit !f }' /etc/fstab; then
		result FAIL nvme "/etc/fstab has no entry UUID=$uuid for $DATA_DIR"
	else
		result PASS nvme "$source (UUID $uuid), $(df -h --output=avail "$DATA_DIR" | tail -n1 | tr -d ' ') free"
	fi
}

check_power() {
	local mode
	mode=$({ nvpmodel -q 2>/dev/null || true; } | sed -n 's/^NV Power Mode: //p')
	if [[ $mode == MAXN* ]]; then
		result PASS power "nvpmodel $mode"
	else
		result FAIL power "nvpmodel mode ${mode:-unknown}, MAXN expected"
	fi
}

check_layout() {
	local spec path want have target
	if [[ $(getent passwd "$APP_USER" | cut -d: -f7) == */nologin ]]; then
		result PASS layout "user $APP_USER (no login shell)"
	else
		result FAIL layout "user $APP_USER missing or with a login shell"
	fi
	for spec in "${LAYOUT_DIRS[@]}" "$ENV_FILE root:$APP_USER $ENV_FILE_MODE"; do
		read -r path want <<<"$spec"
		have=$(stat -c '%U:%G %a' "$path" 2>/dev/null || printf 'missing')
		if [[ $have == "$want" ]]; then
			result PASS layout "$path $have"
		else
			result FAIL layout "$path is $have, $want expected"
		fi
	done
	if [[ -L $OPT_DIR/current ]]; then
		target=$(readlink -f "$OPT_DIR/current")
		if [[ -d $target && $target == "$OPT_DIR"/releases/* ]]; then
			result PASS layout "$OPT_DIR/current -> $target"
		else
			result FAIL layout "$OPT_DIR/current points to $(readlink "$OPT_DIR/current"), not to a release"
		fi
	elif [[ -e $OPT_DIR/current ]]; then
		result FAIL layout "$OPT_DIR/current is not a symlink"
	else
		result WARN layout "$OPT_DIR/current: no release installed yet (release.sh, SV-6.4)"
	fi
}

check_logs() {
	if [[ -f $LOGROTATE_CONF ]] && grep -q 'rotate 14' "$LOGROTATE_CONF" && logrotate --debug "$LOGROTATE_CONF" >/dev/null 2>&1; then
		result PASS logs "$LOGROTATE_CONF: daily, 14 days"
	else
		result FAIL logs "$LOGROTATE_CONF missing, without 'rotate 14', or rejected by logrotate --debug"
	fi
	if grep -q '^MaxRetentionSec=14day' "$JOURNALD_CONF" 2>/dev/null; then
		result PASS logs "journald keeps 14 days ($JOURNALD_CONF)"
	else
		result FAIL logs "journald retention not set ($JOURNALD_CONF)"
	fi
}

check_firewall() {
	local verbose problems
	verbose=$(ufw status verbose 2>&1)
	if ! grep -q '^Status: active' <<<"$verbose"; then
		result FAIL firewall "ufw is not active"
		return
	fi
	if grep -q '^Default: deny (incoming)' <<<"$verbose"; then
		result PASS firewall "active, incoming denied by default"
	else
		result FAIL firewall "incoming traffic is not denied by default: $(grep '^Default:' <<<"$verbose")"
	fi
	problems=$(ufw status | ufw_rules_table | firewall_problems "$HTTP_REDIRECT" "${ADMIN_SUBNETS[@]}")
	if [[ -z $problems ]]; then
		result PASS firewall "open: $(ufw status | ufw_rules_table | awk -F '\t' '$2 ~ /^(ALLOW|LIMIT)/ { print $1 " from " $3 }' |
			sort -u | paste -sd ';' -)"
	else
		while IFS= read -r line; do result FAIL firewall "$line"; done <<<"$problems"
	fi
}

check_ports() {
	local table unexpected
	table=$(ss -H -tulnp 2>/dev/null | listeners_table)
	result INFO ports "network: $(listeners_summary exposed <<<"$table" | paste -sd ';' -)"
	result INFO ports "loopback only: $(listeners_summary loopback <<<"$table" | paste -sd ';' -)"
	unexpected=$(unexpected_exposed <<<"$table")
	if [[ -n $unexpected ]]; then
		result WARN ports "listening beyond loopback (blocked by ufw; bind them to 127.0.0.1): $(paste -sd ';' - <<<"$unexpected")"
	else
		result PASS ports "only expected ports listen beyond loopback ($EXPOSED_OK)"
	fi
}

check_units() {
	local unit enabled active state
	local -a units=("${BASE_UNITS[@]}" "${EXTRA_UNITS[@]}")
	while read -r unit _; do
		[[ -n $unit && " ${units[*]} " != *" $unit "* ]] && units+=("$unit")
	done < <(systemctl list-unit-files --no-legend --state=enabled 'bf-*' caddy.service redis-server.service 2>/dev/null)
	for unit in "${units[@]}"; do
		enabled=$(systemctl is-enabled "$unit" 2>/dev/null)
		active=$(systemctl is-active "$unit" 2>/dev/null)
		if [[ $enabled =~ ^(enabled|static|alias)$ && $active == active ]]; then
			result PASS units "$unit $enabled, $active"
		else
			result FAIL units "$unit is-enabled ${enabled:-unknown}, is-active ${active:-unknown}"
		fi
	done
	state=$(systemctl is-system-running 2>/dev/null)
	case $state in
	running) result PASS units "system running, since $(uptime -s)" ;;
	starting) result WARN units "system still starting (booted $(uptime -s)); run again in a minute" ;;
	*) result FAIL units "system $state; failed: $(systemctl --failed --no-legend --plain | awk '{ print $1 }' | paste -sd ' ' -)" ;;
	esac
}

check_reboot() {
	local boot last_boot="" last_time=""
	boot=$(cat /proc/sys/kernel/random/boot_id)
	[[ -r $STATE_FILE ]] && read -r last_boot last_time <"$STATE_FILE"
	if ((FAILS > 0)); then
		result INFO reboot "boot $boot: not recorded as passing (failed checks)"
		return
	fi
	if ! mountpoint -q "$DATA_DIR"; then
		result WARN reboot "cannot record the boot: $DATA_DIR not mounted"
		return
	fi
	if [[ -n $last_boot && $last_boot != "$boot" ]]; then
		result PASS reboot "every check passes after a reboot (last pass on boot $last_boot at $last_time)"
	elif [[ -z $last_boot ]]; then
		result INFO reboot "first passing run; reboot and run again to check the services come back"
	else
		result INFO reboot "same boot as the last passing run; reboot and run again to check the services come back"
	fi
	mkdir -p "$(dirname "$STATE_FILE")"
	printf '%s %s\n' "$boot" "$(date -Iseconds)" >"$STATE_FILE"
}

need_value() { [[ $# -ge 2 && $2 != -* ]] || usage_error "$1 needs a value"; }

usage_error() {
	printf 'verify_base.sh: %s\n\n' "$*" >&2
	usage >&2
	exit 2
}

parse_args() {
	MAX_OFFSET_MS=100
	ADMIN_SUBNETS=()
	EXTRA_UNITS=()
	HTTP_REDIRECT=1
	while (($#)); do
		case $1 in
		--max-offset-ms)
			need_value "$@"
			[[ $2 =~ ^[0-9]+([.][0-9]+)?$ ]] || usage_error "--max-offset-ms: a number of milliseconds: $2"
			MAX_OFFSET_MS=$2
			shift 2
			;;
		--admin-subnet)
			need_value "$@"
			ADMIN_SUBNETS+=("$2")
			shift 2
			;;
		--unit)
			need_value "$@"
			EXTRA_UNITS+=("$2")
			shift 2
			;;
		--no-http)
			HTTP_REDIRECT=0
			shift
			;;
		-h | --help)
			usage
			exit 0
			;;
		*) usage_error "unknown argument: $1" ;;
		esac
	done
}

main() {
	parse_args "$@"
	if [[ $EUID -ne 0 ]]; then
		printf 'verify_base.sh: run as root (sudo): ufw and the socket owners need it\n' >&2
		exit 1
	fi
	printf 'BF-02 Jetson base system, %s, boot %s\n' "$(date -Iseconds)" "$(cat /proc/sys/kernel/random/boot_id)"
	check_platform
	check_clock
	check_nvme
	check_power
	check_layout
	check_logs
	check_firewall
	check_ports
	check_units
	check_reboot
	printf '\n%d pass, %d warn, %d fail\n' "$PASSES" "$WARNS" "$FAILS"
	((FAILS == 0))
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
