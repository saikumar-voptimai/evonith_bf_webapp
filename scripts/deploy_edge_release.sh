#!/usr/bin/env bash
# Deploy one already-tested Git commit to an Evonith edge device.
set -Eeuo pipefail

repo_dir="${EVONITH_DEPLOY_REPO_DIR:-/opt/evonith-bf}"
target_ref="${EVONITH_DEPLOY_REF:-${GITHUB_SHA:-}}"
deploy_branch="${EVONITH_DEPLOY_BRANCH:-release}"
device="${EVONITH_EDGE_DEVICE_TYPE:-auto}"
python_bin="${EVONITH_DEPLOY_PYTHON:-/usr/bin/python3}"
uv_bin="${EVONITH_UV_BIN:-uv}"
service_name="${EVONITH_BACKEND_SERVICE:-evonith-backend}"
health_url="${EVONITH_DEPLOY_HEALTH_URL:-http://127.0.0.1:1432/api/v1/health}"
readiness_url="${EVONITH_DEPLOY_READINESS_URL:-http://127.0.0.1:1432/api/v1/readiness}"
health_attempts="${EVONITH_DEPLOY_HEALTH_ATTEMPTS:-30}"
dry_run=0

usage() {
  cat <<'EOF'
Usage: scripts/deploy_edge_release.sh --ref <commit> --device <device> [options]

Options:
  --ref <commit>       Exact tested Git commit to deploy (required)
  --branch <branch>    Remote branch that must contain the commit (default: release)
  --device <device>    jetson, raspberry-pi, generic-arm64, or auto
  --repo-dir <path>    Production checkout (default: /opt/evonith-bf)
  --python <path>      System Python used by uv (default: /usr/bin/python3)
  --dry-run            Validate arguments and print actions without changing anything
  --help               Show this help

The script refuses a dirty production checkout, deploys only a commit contained
in origin/<branch>, verifies device compatibility, restarts systemd, checks
health/readiness, and automatically returns to the previous commit on failure.
EOF
}

while (($#)); do
  case "$1" in
    --ref) target_ref="${2:?missing value for --ref}"; shift 2 ;;
    --branch) deploy_branch="${2:?missing value for --branch}"; shift 2 ;;
    --device) device="${2:?missing value for --device}"; shift 2 ;;
    --repo-dir) repo_dir="${2:?missing value for --repo-dir}"; shift 2 ;;
    --python) python_bin="${2:?missing value for --python}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$target_ref" ]]; then
  echo "--ref is required (or set EVONITH_DEPLOY_REF/GITHUB_SHA)" >&2
  exit 2
fi
if ! git check-ref-format --branch "$deploy_branch" >/dev/null 2>&1; then
  echo "Invalid deployment branch: $deploy_branch" >&2
  exit 2
fi
if [[ ! "$health_attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVONITH_DEPLOY_HEALTH_ATTEMPTS must be a positive integer" >&2
  exit 2
fi

log() {
  printf '[evonith-deploy] %s\n' "$*"
}

run() {
  if ((dry_run)); then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

if ((dry_run)); then
  log "device=$device branch=$deploy_branch ref=$target_ref repo=$repo_dir"
  run git -C "$repo_dir" fetch --prune origin \
    "+refs/heads/$deploy_branch:refs/remotes/origin/$deploy_branch"
  run "$uv_bin" sync --python "$python_bin" --no-dev --group edge --locked
  run sudo systemctl restart "$service_name"
  run curl --fail --silent --show-error "$health_url"
  run curl --fail --silent --show-error "$readiness_url"
  exit 0
fi

for command in git curl flock sudo; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command is unavailable: $command" >&2
    exit 1
  fi
done
if ! command -v "$uv_bin" >/dev/null 2>&1; then
  echo "uv is unavailable: $uv_bin" >&2
  exit 1
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Python interpreter is unavailable: $python_bin" >&2
  exit 1
fi
if [[ ! -d "$repo_dir/.git" ]]; then
  echo "Production checkout is missing: $repo_dir" >&2
  exit 1
fi

exec 9>"${TMPDIR:-/tmp}/evonith-edge-deploy.lock"
if ! flock -n 9; then
  echo "Another edge deployment is already running" >&2
  exit 1
fi

cd "$repo_dir"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Production checkout is dirty; refusing automatic deployment" >&2
  git status --short >&2
  exit 1
fi

previous_ref="$(git rev-parse HEAD)"
target_sha=""
rollback_armed=0
rollback_running=0

wait_for_endpoint() {
  local url="$1"
  local label="$2"
  local attempt
  for ((attempt = 1; attempt <= health_attempts; attempt++)); do
    if curl --fail --silent --show-error --max-time 5 "$url" >/dev/null; then
      log "$label passed on attempt $attempt"
      return 0
    fi
    sleep 1
  done
  log "$label failed after $health_attempts attempts"
  return 1
}

restore_previous_release() {
  local original_status="$1"
  if ((rollback_running)) || ((rollback_armed == 0)); then
    return "$original_status"
  fi
  rollback_running=1
  trap - ERR
  log "deployment failed; rolling back to $previous_ref"
  git switch --detach "$previous_ref" || true
  "$uv_bin" sync --python "$python_bin" --no-dev --group edge --locked || true
  sudo systemctl restart "$service_name" || true
  wait_for_endpoint "$health_url" "rollback health" || true
  log "rollback attempt finished; inspect service logs before retrying"
  return "$original_status"
}

on_error() {
  local status=$?
  restore_previous_release "$status"
  exit "$status"
}
trap on_error ERR

log "fetching origin/$deploy_branch"
git fetch --prune origin \
  "+refs/heads/$deploy_branch:refs/remotes/origin/$deploy_branch"
target_sha="$(git rev-parse --verify "${target_ref}^{commit}")"
if ! git merge-base --is-ancestor "$target_sha" "origin/$deploy_branch"; then
  echo "Ref $target_sha is not contained in origin/$deploy_branch" >&2
  exit 1
fi
if [[ "$target_sha" == "$previous_ref" ]]; then
  log "commit $target_sha is already deployed; verifying service only"
  sudo systemctl is-active --quiet "$service_name"
  wait_for_endpoint "$health_url" "health"
  wait_for_endpoint "$readiness_url" "readiness"
  trap - ERR
  exit 0
fi

rollback_armed=1
log "switching from $previous_ref to tested commit $target_sha"
git switch --detach "$target_sha"

"$python_bin" scripts/verify_edge_device.py --expect "$device" --json
"$uv_bin" sync --python "$python_bin" --no-dev --group edge --locked
.venv/bin/python -c "import fastapi, pandas, sklearn, furnace_data; print('backend imports: OK')"

log "restarting $service_name"
sudo systemctl restart "$service_name"
sudo systemctl is-active --quiet "$service_name"
wait_for_endpoint "$health_url" "health"
wait_for_endpoint "$readiness_url" "readiness"

rollback_armed=0
trap - ERR
log "deployment successful device=$device commit=$target_sha"
