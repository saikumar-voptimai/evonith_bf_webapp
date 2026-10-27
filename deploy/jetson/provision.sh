#!/usr/bin/env bash
# BF-02 webapp: base system of the Jetson AGX Orin (SV-6.1, #315). Run as root on the Orin; idempotent: every step
# reads the current state and changes only what differs, so a second run reports "no changes".
#
#   sudo deploy/jetson/provision.sh --nvme-uuid <UUID> --ntp-server <host> --admin-subnet <CIDR> [options]
#
# Steps, in this order (main):
#   platform  JetPack 6.x (L4T R36), Ubuntu 22.04, aarch64; refuses anything else
#   packages  chrony, ufw, logrotate through apt, only the missing ones
#   user      system user bfapp: no login shell, home /var/lib/bf-webapp, groups video and render (GPU)
#   nvme      the NVMe filesystem (ext4 or xfs, formatted by hand: README) in /etc/fstab by UUID, at /var/lib/bf-webapp
#   layout    /opt/bf-webapp/releases (+ current, the symlink release.sh sets), /var/lib/bf-webapp/{storage,cache},
#             /var/log/bf-webapp, /etc/bf-webapp/env (root:bfapp 0640; created once, never rewritten)
#   power     nvpmodel MAXN now and at every boot, then jetson_clocks (bf-jetson-power.service)
#   clock     chrony against the plant NTP server(s) only
#   logs      logrotate: /var/log/bf-webapp/*.log daily, 14 days; journald keeps 14 days
#   firewall  ufw: incoming denied except 443/tcp and 80/tcp (Caddy's HTTPS redirect) from anywhere, 22/tcp from the
#             admin subnet(s); refuses to run while an SSH client outside those subnets is connected
#
# The render_* functions and the parsers are pure: tests/test_jetson_scripts.py sources this file and calls them.
set -euo pipefail

APP_USER=bfapp
OPT_DIR=/opt/bf-webapp
DATA_DIR=/var/lib/bf-webapp
LOG_DIR=/var/log/bf-webapp
ETC_DIR=/etc/bf-webapp
ENV_FILE=$ETC_DIR/env
ENV_FILE_MODE=640
POWER_UNIT=bf-jetson-power.service
FSTAB_MARK='# bf-webapp data disk (deploy/jetson/provision.sh, SV-6.1)'
CHRONY_CONF=/etc/chrony/chrony.conf
LOGROTATE_CONF=/etc/logrotate.d/bf-webapp
JOURNALD_CONF=/etc/systemd/journald.conf.d/bf-webapp.conf
LOG_RETENTION_DAYS=14
PACKAGES=(chrony ufw logrotate)
# path owner:group mode; verify_base.sh holds the same table (tested)
LAYOUT_DIRS=(
	"/opt/bf-webapp root:root 755"
	"/opt/bf-webapp/releases root:root 755"
	"/var/lib/bf-webapp bfapp:bfapp 750"
	"/var/lib/bf-webapp/storage bfapp:bfapp 750"
	"/var/lib/bf-webapp/cache bfapp:bfapp 750"
	"/var/log/bf-webapp bfapp:bfapp 750"
	"/etc/bf-webapp root:bfapp 750"
)

CHANGES=()
REBOOT_REASONS=()
WROTE=0

usage() {
	cat <<'EOF'
Usage: provision.sh --nvme-uuid UUID --ntp-server HOST [--ntp-server HOST ...]
                    --admin-subnet CIDR [--admin-subnet CIDR ...] [--no-http] [--allow-ssh-lockout]

Base system of the BF-02 Jetson (SV-6.1). Run as root; safe to run again.

  --nvme-uuid UUID      filesystem UUID of the NVMe data partition (blkid), mounted at /var/lib/bf-webapp
  --ntp-server HOST     plant NTP server (repeat for more); chrony uses only these
  --admin-subnet CIDR   IPv4 network allowed to reach SSH (22/tcp), e.g. 10.20.30.0/24 (repeat for more)
  --no-http             keep 80/tcp closed (no HTTP-to-HTTPS redirect for clients typing http://)
  --allow-ssh-lockout   enable the firewall although an SSH client outside the admin subnets is connected
  -h, --help            this text
EOF
}

note() { printf '   %s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
die() {
	printf 'provision.sh: %s\n' "$*" >&2
	exit 1
}
usage_error() {
	printf 'provision.sh: %s\n\n' "$*" >&2
	usage >&2
	exit 2
}
changed() {
	CHANGES+=("$*")
	note "changed: $*"
}

# ---- pure helpers (tested) ------------------------------------------------------------------------------------------

is_uuid() {
	local re='^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
	[[ $1 =~ $re ]]
}

is_host() { # host name, IPv4 or IPv6 address
	local name='^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$' ipv6='^[0-9A-Fa-f:]*:[0-9A-Fa-f:]*$'
	[[ $1 =~ $name || $1 =~ $ipv6 ]]
}

ipv4_to_int() {
	local re='^([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$' a b c d
	[[ $1 =~ $re ]] || return 1
	a=$((10#${BASH_REMATCH[1]})) b=$((10#${BASH_REMATCH[2]})) c=$((10#${BASH_REMATCH[3]})) d=$((10#${BASH_REMATCH[4]}))
	((a <= 255 && b <= 255 && c <= 255 && d <= 255)) || return 1
	printf '%d\n' $(((a << 24) | (b << 16) | (c << 8) | d))
}

int_to_ipv4() { printf '%d.%d.%d.%d\n' $(($1 >> 24 & 255)) $(($1 >> 16 & 255)) $(($1 >> 8 & 255)) $(($1 & 255)); }

prefix_mask() { printf '%d\n' $(((0xFFFFFFFF << (32 - $1)) & 0xFFFFFFFF)); }

# "10.20.30.7/24" -> "10.20.30.0/24"; prefixes /8 to /32 only (SSH from everyone is never an admin subnet)
normalize_cidr() {
	local re='^([0-9.]+)/([0-9]{1,2})$' n prefix
	[[ $1 =~ $re ]] || return 1
	prefix=$((10#${BASH_REMATCH[2]}))
	((prefix >= 8 && prefix <= 32)) || return 1
	n=$(ipv4_to_int "${BASH_REMATCH[1]}") || return 1
	printf '%s/%d\n' "$(int_to_ipv4 $((n & $(prefix_mask "$prefix"))))" "$prefix"
}

ip_in_cidr() {
	local n net mask
	n=$(ipv4_to_int "$1") || return 1
	net=$(ipv4_to_int "${2%/*}") || return 1
	mask=$(prefix_mask "${2#*/}")
	(((n & mask) == (net & mask)))
}

# stdin: `ss -Htn state established '( sport = :22 )'`; stdout: one client address per line
ssh_peer_addresses() {
	local line peer
	while read -r line; do
		[[ -n $line ]] || continue
		peer=${line##* }
		peer=${peer%:*}
		peer=${peer#[}
		peer=${peer%]}
		peer=${peer#::ffff:}
		printf '%s\n' "$peer"
	done
}

# stdin: client addresses; args: admin subnets; stdout: the clients a firewall with only those subnets on 22 locks out
peers_outside() {
	local peer cidr inside
	while read -r peer; do
		[[ -n $peer ]] || continue
		inside=0
		for cidr in "$@"; do
			if ip_in_cidr "$peer" "$cidr"; then
				inside=1
				break
			fi
		done
		((inside)) || printf '%s\n' "$peer"
	done
}

# stdin: /etc/nv_tegra_release; stdout: L4T version, e.g. "36.4.3"
l4t_version() {
	local line re='^# R([0-9]+) \(release\), REVISION: ([0-9]+)\.([0-9]+)'
	IFS= read -r line || [[ -n $line ]] || return 1
	[[ $line =~ $re ]] || return 1
	printf '%s.%s.%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
}

jetpack6() { [[ ${1%%.*} == 36 ]]; } # JetPack 6.0 to 6.2 ship L4T R36.x

# stdin: /etc/nvpmodel.conf; stdout: "<id> <name>" of the MAXN mode (MAXN, else the first MAXN_* mode)
maxn_mode() {
	awk '
		/^< POWER_MODEL / {
			id = ""; name = ""
			for (i = 1; i <= NF; i++) {
				if ($i ~ /^ID=/) id = substr($i, 4)
				if ($i ~ /^NAME=/) name = substr($i, 6)
			}
			if (name == "MAXN") { print id " " name; found = 1; exit }
			if (other == "" && name ~ /^MAXN/) other = id " " name
		}
		END { if (!found) { if (other == "") exit 1; print other } }'
}

render_fstab_line() { # uuid fstype
	printf 'UUID=%s %s %s defaults,noatime,nofail,x-systemd.device-timeout=30s 0 2\n' "$1" "$DATA_DIR" "$2"
}

# stdin: /etc/fstab; $1: the data-disk line; stdout: fstab with that line as the only entry for the data directory
fstab_with_entry() {
	awk -v mp="$DATA_DIR" -v mark="$FSTAB_MARK" -v entry="$1" '
		$0 == mark { next }
		$0 !~ /^[[:space:]]*#/ && $2 == mp { next }
		{ print }
		END { print mark; print entry }'
}

render_chrony_conf() { # servers...
	printf '# BF-02 webapp (deploy/jetson/provision.sh, SV-6.1): time from the plant NTP server only. Rewritten on every run.\n'
	local server
	for server in "$@"; do
		printf 'server %s iburst\n' "$server"
	done
	cat <<'EOF'
keyfile /etc/chrony/chrony.keys
driftfile /var/lib/chrony/chrony.drift
logdir /var/log/chrony
maxupdateskew 100.0
# Step the clock when it is more than 1 s off during the first three updates (the clock can be wrong after a boot).
makestep 1 3
rtcsync
leapsectz right/UTC
EOF
}

render_power_unit() { # nvpmodel-binary mode-id jetson_clocks-binary
	cat <<EOF
[Unit]
Description=BF-02 webapp: nvpmodel MAXN and jetson_clocks (deploy/jetson/provision.sh, SV-6.1)
After=nvpmodel.service

[Service]
Type=oneshot
RemainAfterExit=yes
# "-": a mode change that needs a reboot must not stop jetson_clocks; verify_base.sh checks the mode.
ExecStart=-$1 -m $2
ExecStart=$3

[Install]
WantedBy=multi-user.target
EOF
}

render_logrotate() {
	cat <<EOF
# BF-02 webapp (deploy/jetson/provision.sh, SV-6.1): ${LOG_RETENTION_DAYS} days of daily logs.
$LOG_DIR/*.log {
	su $APP_USER $APP_USER
	daily
	rotate $LOG_RETENTION_DAYS
	maxage $LOG_RETENTION_DAYS
	missingok
	notifempty
	compress
	delaycompress
	copytruncate
	dateext
}
EOF
}

render_journald() {
	cat <<EOF
# BF-02 webapp (deploy/jetson/provision.sh, SV-6.1): service output kept ${LOG_RETENTION_DAYS} days, across reboots.
[Journal]
Storage=persistent
MaxRetentionSec=${LOG_RETENTION_DAYS}day
EOF
}

render_env_template() { # jwt-secret
	cat <<EOF
# BF-02 webapp environment: EnvironmentFile= of the services (SV-6.3). root:$APP_USER 0$ENV_FILE_MODE.
# Created once by deploy/jetson/provision.sh (SV-6.1), never rewritten by it. KEY=value lines, no "export".
BF_API_ENVIRONMENT=prod
BF_API_JWT_SECRET=$1
BF_API_COOKIE_SECURE=true
BF_REDIS_URL=redis://127.0.0.1:6379/0
UV_CACHE_DIR=$DATA_DIR/cache/uv
XDG_CACHE_HOME=$DATA_DIR/cache
# Fill in before the first release (CLAUDE.md, "Environment Variables"):
# DATABASE_URL=postgresql://...
# INFLUX_ONLINE_TOKEN=
# INFLUX_OFFLINE_TOKEN=
# OPENROUTER_API_KEY=
# OPENAI_API_KEY=
# QDRANT_URL=
# QDRANT_API_KEY=
EOF
}

# stdout: "<ufw rule words><TAB><comment>" per rule, in the order they are added
ufw_rules() { # http(0|1) subnets...
	local http=$1 cidr
	shift
	printf 'allow 443/tcp\tbf-webapp-https\n'
	if ((http)); then printf 'allow 80/tcp\tbf-webapp-http-redirect\n'; fi
	for cidr in "$@"; do
		printf 'allow from %s to any port 22 proto tcp\tbf-webapp-ssh-admin\n' "$cidr"
	done
}

# stdin: ufw_rules; stdout: the same rules as `ufw show added` lists them
ufw_show_form() {
	local rule comment
	while IFS=$'\t' read -r rule comment; do
		printf "ufw %s comment '%s'\n" "$rule" "$comment"
	done
}

# ---- system helpers -------------------------------------------------------------------------------------------------

# write_file PATH MODE OWNER:GROUP < content; WROTE=1 when the content changed (owner and mode are set either way)
write_file() {
	local path=$1 mode=$2 owner=$3 tmp
	WROTE=0
	mkdir -p "$(dirname "$path")"
	tmp=$(mktemp "$(dirname "$path")/.$(basename "$path").XXXXXX") # hidden: never read as a config while it exists
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

backup_once() { [[ ! -f $1 || -e $1.bf-orig ]] || cp -p "$1" "$1.bf-orig"; }

ensure_dir() { # path owner:group mode
	local have=""
	[[ -d $1 ]] && have=$(stat -c '%U:%G %a' "$1")
	if [[ $have != "$2 $3" ]]; then
		mkdir -p "$1"
		chown "$2" "$1"
		chmod "$3" "$1"
		changed "$1 ($2 $3)"
	fi
}

nvpmodel_current() { { nvpmodel -q 2>/dev/null || true; } | sed -n 's/^NV Power Mode: //p'; }

# ---- steps ----------------------------------------------------------------------------------------------------------

step_platform() {
	step "platform: JetPack 6.x"
	local l4t model jetpack ubuntu
	[[ $(uname -m) == aarch64 ]] || die "machine is $(uname -m), not aarch64: run this on the Jetson"
	[[ -r /etc/nv_tegra_release ]] || die "/etc/nv_tegra_release is missing: not a JetPack (L4T) system"
	l4t=$(l4t_version </etc/nv_tegra_release) || die "cannot read the L4T release from /etc/nv_tegra_release"
	jetpack6 "$l4t" || die "L4T $l4t is not JetPack 6.x (L4T R36.x); flash JetPack 6 first"
	ubuntu=$(. /etc/os-release && printf '%s' "$VERSION_ID")
	[[ $ubuntu == 22.04 ]] || die "Ubuntu $ubuntu; JetPack 6 is Ubuntu 22.04"
	model=$(tr -d '\0' 2>/dev/null </proc/device-tree/model || printf 'unknown model')
	jetpack=$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null || printf 'meta-package not installed')
	note "$model; L4T $l4t; Ubuntu $ubuntu; nvidia-jetpack $jetpack"
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

step_user() {
	step "user: $APP_USER"
	local group
	if ! id -u "$APP_USER" >/dev/null 2>&1; then
		useradd --system --user-group --home-dir "$DATA_DIR" --no-create-home --shell /usr/sbin/nologin "$APP_USER"
		changed "system user $APP_USER"
	fi
	for group in video render; do
		if getent group "$group" >/dev/null && [[ " $(id -nG "$APP_USER") " != *" $group "* ]]; then
			usermod -aG "$group" "$APP_USER"
			changed "$APP_USER in group $group"
		fi
	done
	note "$(id "$APP_USER")"
}

step_nvme() {
	step "NVMe data disk at $DATA_DIR"
	local dev fstype entry
	dev=$(blkid -U "$NVME_UUID" 2>/dev/null) ||
		die "no filesystem with UUID $NVME_UUID: partition and format the NVMe first (README, 'Base system')"
	[[ $dev == /dev/nvme* ]] || die "UUID $NVME_UUID is on $dev, not on an NVMe device"
	fstype=$(blkid -s TYPE -o value "$dev")
	[[ $fstype == ext4 || $fstype == xfs ]] || die "$dev holds '$fstype'; ext4 or xfs expected"
	if awk -v u="UUID=$NVME_UUID" -v mp="$DATA_DIR" '$0 !~ /^[[:space:]]*#/ && $1 == u && $2 != mp { f = 1 } END { exit !f }' \
		/etc/fstab; then
		die "/etc/fstab already mounts UUID $NVME_UUID elsewhere; remove that entry first"
	fi
	if mountpoint -q "$DATA_DIR"; then
		[[ $(findmnt -n -o UUID --mountpoint "$DATA_DIR") == "$NVME_UUID" ]] ||
			die "$DATA_DIR is mounted from $(findmnt -n -o SOURCE --mountpoint "$DATA_DIR"), not from UUID $NVME_UUID"
	elif [[ -d $DATA_DIR && -n $(ls -A "$DATA_DIR") ]]; then
		die "$DATA_DIR holds files on the root filesystem; move them to the NVMe, then run again"
	fi
	entry=$(render_fstab_line "$NVME_UUID" "$fstype")
	backup_once /etc/fstab
	write_file /etc/fstab 644 root:root < <(fstab_with_entry "$entry" </etc/fstab)
	if ((WROTE)); then
		systemctl daemon-reload
		changed "/etc/fstab: $entry"
	fi
	if mountpoint -q "$DATA_DIR"; then
		note "mounted: $dev ($fstype)"
	else
		mkdir -p "$DATA_DIR"
		mount "$DATA_DIR"
		changed "mounted $dev ($fstype) at $DATA_DIR"
	fi
}

step_layout() {
	step "layout"
	local spec path owner mode secret
	for spec in "${LAYOUT_DIRS[@]}"; do
		read -r path owner mode <<<"$spec"
		ensure_dir "$path" "$owner" "$mode"
	done
	if [[ -L $OPT_DIR/current ]]; then
		note "current -> $(readlink "$OPT_DIR/current")"
	elif [[ -e $OPT_DIR/current ]]; then
		die "$OPT_DIR/current exists and is not a symlink; release.sh (SV-6.4) owns it"
	else
		note "current: no release yet (release.sh, SV-6.4, points it at releases/<tag>)"
	fi
	if [[ ! -e $ENV_FILE ]]; then
		secret=$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')
		write_file "$ENV_FILE" "$ENV_FILE_MODE" "root:$APP_USER" < <(render_env_template "$secret")
		changed "$ENV_FILE (template with a new BF_API_JWT_SECRET)"
	elif [[ $(stat -c '%U:%G %a' "$ENV_FILE") != "root:$APP_USER $ENV_FILE_MODE" ]]; then
		chown "root:$APP_USER" "$ENV_FILE"
		chmod "$ENV_FILE_MODE" "$ENV_FILE"
		changed "$ENV_FILE owner and mode (content kept)"
	else
		note "$ENV_FILE kept"
	fi
}

step_power() {
	step "power: nvpmodel MAXN and jetson_clocks at boot"
	local nvp jc mode id name current
	nvp=$(command -v nvpmodel) || die "nvpmodel not found (JetPack package nvidia-l4t-nvpmodel)"
	jc=$(command -v jetson_clocks) || die "jetson_clocks not found (JetPack package nvidia-l4t-tools)"
	mode=$(maxn_mode </etc/nvpmodel.conf) || die "/etc/nvpmodel.conf has no MAXN mode"
	read -r id name <<<"$mode"
	current=$(nvpmodel_current)
	if [[ $current != "$name" ]]; then
		"$nvp" -m "$id" </dev/null || true # answers "no" when nvpmodel asks for a reboot
		changed "nvpmodel mode $id ($name), was ${current:-unknown}"
		[[ $(nvpmodel_current) == "$name" ]] || REBOOT_REASONS+=("nvpmodel switches to $name at the next boot")
	fi
	write_file "/etc/systemd/system/$POWER_UNIT" 644 root:root < <(render_power_unit "$nvp" "$id" "$jc")
	if ((WROTE)); then
		systemctl daemon-reload
		changed "$POWER_UNIT"
	fi
	if ! systemctl is-enabled --quiet "$POWER_UNIT"; then
		systemctl enable "$POWER_UNIT"
		changed "$POWER_UNIT enabled"
	fi
	if ((WROTE)) || ! systemctl is-active --quiet "$POWER_UNIT"; then systemctl restart "$POWER_UNIT"; fi
	note "nvpmodel: $(nvpmodel_current); $POWER_UNIT $(systemctl is-active "$POWER_UNIT")"
}

step_clock() {
	step "clock: chrony against ${NTP_SERVERS[*]}"
	local restart=0
	backup_once "$CHRONY_CONF"
	write_file "$CHRONY_CONF" 644 root:root < <(render_chrony_conf "${NTP_SERVERS[@]}")
	if ((WROTE)); then
		restart=1
		changed "$CHRONY_CONF"
	fi
	if systemctl is-enabled --quiet systemd-timesyncd.service 2>/dev/null; then
		systemctl disable --now systemd-timesyncd.service
		changed "systemd-timesyncd disabled (chrony keeps the time)"
	fi
	if ! systemctl is-enabled --quiet chrony.service; then
		systemctl enable chrony.service
		changed "chrony enabled"
	fi
	if ((restart)) || ! systemctl is-active --quiet chrony.service; then systemctl restart chrony.service; fi
	if chronyc waitsync 12 0.1 0 5 >/dev/null 2>&1; then
		note "synchronised: $(chronyc tracking | sed -n 's/^System time *: //p')"
	else
		note "not synchronised within 60 s: check that ${NTP_SERVERS[*]} answers on 123/udp; verify_base.sh checks again"
	fi
}

step_logs() {
	step "logs: logrotate and journald, $LOG_RETENTION_DAYS days"
	write_file "$LOGROTATE_CONF" 644 root:root < <(render_logrotate)
	if ((WROTE)); then changed "$LOGROTATE_CONF"; fi
	logrotate --debug "$LOGROTATE_CONF" >/dev/null 2>&1 || die "logrotate rejects $LOGROTATE_CONF (logrotate --debug)"
	write_file "$JOURNALD_CONF" 644 root:root < <(render_journald)
	if ((WROTE)); then
		mkdir -p /var/log/journal
		systemctl restart systemd-journald.service
		changed "$JOURNALD_CONF"
	fi
	if ! systemctl is-enabled --quiet logrotate.timer; then
		systemctl enable --now logrotate.timer
		changed "logrotate.timer enabled"
	fi
}

step_firewall() {
	step "firewall: ufw"
	local outside want have status rule comment
	local -a words
	outside=$({ ss -Htn state established '( sport = :22 )' 2>/dev/null || true; } | ssh_peer_addresses |
		peers_outside "${ADMIN_SUBNETS[@]}")
	if [[ -n $outside ]] && ((!ALLOW_SSH_LOCKOUT)); then
		die "SSH client(s) $(tr '\n' ' ' <<<"$outside")outside ${ADMIN_SUBNETS[*]} would lose SSH; connect from an admin subnet, or pass --allow-ssh-lockout"
	fi
	want=$(ufw_rules "$HTTP_REDIRECT" "${ADMIN_SUBNETS[@]}" | ufw_show_form)
	have=$({ ufw show added 2>/dev/null || true; } | sed 1d)
	status=$(ufw status verbose)
	if [[ $want == "$have" ]] && grep -q '^Status: active' <<<"$status" &&
		grep -q '^Default: deny (incoming), allow (outgoing), deny (routed)' <<<"$status"; then
		note "rules unchanged"
	else
		ufw --force reset >/dev/null
		ufw default deny incoming >/dev/null
		ufw default allow outgoing >/dev/null
		ufw default deny routed >/dev/null
		while IFS=$'\t' read -r rule comment; do
			read -r -a words <<<"$rule"
			ufw "${words[@]}" comment "$comment" >/dev/null
		done < <(ufw_rules "$HTTP_REDIRECT" "${ADMIN_SUBNETS[@]}")
		ufw --force enable >/dev/null
		changed "ufw rules: $(tr '\n' ';' <<<"$want")"
	fi
	if ! systemctl is-enabled --quiet ufw.service; then
		systemctl enable ufw.service
		changed "ufw.service enabled"
	fi
	ufw status | sed 's/^/   /'
}

summary() {
	step "summary"
	local reason
	if ((${#CHANGES[@]})); then note "${#CHANGES[@]} change(s)"; else note "no changes: the system already matched"; fi
	for reason in "${REBOOT_REASONS[@]}"; do note "reboot required: $reason"; done
	note "next: sudo deploy/jetson/verify_base.sh; then reboot and run it again (services back after a reboot)"
}

need_value() { [[ $# -ge 2 && $2 != -* ]] || usage_error "$1 needs a value"; }

parse_args() {
	NVME_UUID=""
	NTP_SERVERS=()
	ADMIN_SUBNETS=()
	HTTP_REDIRECT=1
	ALLOW_SSH_LOCKOUT=0
	local raw=() item i
	while (($#)); do
		case $1 in
		--nvme-uuid)
			need_value "$@"
			NVME_UUID=${2,,}
			shift 2
			;;
		--ntp-server)
			need_value "$@"
			NTP_SERVERS+=("$2")
			shift 2
			;;
		--admin-subnet)
			need_value "$@"
			raw+=("$2")
			shift 2
			;;
		--no-http)
			HTTP_REDIRECT=0
			shift
			;;
		--allow-ssh-lockout)
			ALLOW_SSH_LOCKOUT=1
			shift
			;;
		-h | --help)
			usage
			exit 0
			;;
		*) usage_error "unknown argument: $1" ;;
		esac
	done
	[[ -n $NVME_UUID ]] || usage_error "--nvme-uuid is required"
	is_uuid "$NVME_UUID" || usage_error "--nvme-uuid: not a filesystem UUID (8-4-4-4-12 hex): $NVME_UUID"
	((${#NTP_SERVERS[@]})) || usage_error "--ntp-server is required (the plant NTP server)"
	for item in "${NTP_SERVERS[@]}"; do
		is_host "$item" || usage_error "--ntp-server: not a host name or address: $item"
	done
	((${#raw[@]})) || usage_error "--admin-subnet is required (the network allowed to reach SSH)"
	for i in "${!raw[@]}"; do
		ADMIN_SUBNETS[i]=$(normalize_cidr "${raw[i]}") ||
			usage_error "--admin-subnet: IPv4 network in CIDR form, /8 to /32: ${raw[i]}"
	done
}

main() {
	parse_args "$@"
	[[ $EUID -eq 0 ]] || die "run as root (sudo)"
	step_platform
	step_packages
	step_user
	step_nvme
	step_layout
	step_power
	step_clock
	step_logs
	step_firewall
	summary
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
