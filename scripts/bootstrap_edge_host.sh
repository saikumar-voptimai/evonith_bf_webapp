#!/usr/bin/env bash
# Prepare a Debian/Ubuntu ARM64 host for an Evonith backend installation.
set -euo pipefail

device=""
repo_dir="/opt/evonith-bf"
runtime_dir="/var/lib/evonith-bf"
config_dir="/etc/evonith-bf"
service_user="evonith"
deploy_owner="${SUDO_USER:-${USER:-root}}"
dry_run=0

usage() {
  cat <<'EOF'
Usage: sudo scripts/bootstrap_edge_host.sh --device <jetson|raspberry-pi> [options]

Options:
  --device <name>       Required device profile
  --repo-dir <path>     Source checkout (default: /opt/evonith-bf)
  --deploy-owner <user> User that owns and updates the source checkout
  --dry-run             Print privileged actions without applying them
  --help                Show this help

This script creates the service account and protected directories, installs a
device-specific environment template when no environment file exists, and
installs the systemd unit. It never starts the backend and never overwrites an
existing /etc/evonith-bf/backend.env.
EOF
}

while (($#)); do
  case "$1" in
    --device) device="${2:?missing value for --device}"; shift 2 ;;
    --repo-dir) repo_dir="${2:?missing value for --repo-dir}"; shift 2 ;;
    --deploy-owner) deploy_owner="${2:?missing value for --deploy-owner}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$device" in
  jetson) env_template="infra/env/edge.jetson.env.example" ;;
  raspberry-pi|pi) device="raspberry-pi"; env_template="infra/env/edge.raspberry-pi.env.example" ;;
  *) echo "--device must be jetson or raspberry-pi" >&2; exit 2 ;;
esac

if [[ ! -f "$repo_dir/$env_template" ]] || [[ ! -f "$repo_dir/infra/systemd/evonith-backend.service.example" ]]; then
  echo "Run this after cloning the repository into $repo_dir" >&2
  exit 1
fi
if ! id "$deploy_owner" >/dev/null 2>&1; then
  echo "Deployment owner does not exist: $deploy_owner" >&2
  exit 1
fi
if ((dry_run == 0)) && [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "Run with sudo, or use --dry-run" >&2
  exit 1
fi

run() {
  if ((dry_run)); then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

if ! id "$service_user" >/dev/null 2>&1; then
  run useradd --system --create-home --home-dir "/var/lib/$service_user" \
    --shell /usr/sbin/nologin "$service_user"
fi

if [[ "$device" == "jetson" ]]; then
  getent group video >/dev/null && run usermod --append --groups video "$service_user"
  getent group render >/dev/null && run usermod --append --groups render "$service_user"
fi

run install -d -o "$deploy_owner" -g "$deploy_owner" -m 0755 "$repo_dir"
run install -d -o "$service_user" -g "$service_user" -m 0750 "$runtime_dir"
run install -d -o root -g "$service_user" -m 0750 "$config_dir"

if [[ ! -e "$config_dir/backend.env" ]]; then
  run install -o root -g "$service_user" -m 0640 \
    "$repo_dir/$env_template" "$config_dir/backend.env"
else
  echo "Keeping existing protected configuration: $config_dir/backend.env"
fi

run install -o root -g root -m 0644 \
  "$repo_dir/infra/systemd/evonith-backend.service.example" \
  /etc/systemd/system/evonith-backend.service
run systemctl daemon-reload

echo "Host bootstrap complete for $device."
echo "Next: edit $config_dir/backend.env, install dependencies, validate, then enable the service."
