#!/usr/bin/env bash
# One CI step (SV-0.11, #280): runs the command with its output in the log as usual; when it fails on GitHub Actions,
# the last 50 lines are repeated as an error annotation, readable on the pull request and through the Checks API
# without opening the log.
#
#   bash scripts/ci/step.sh "<title>" <command> [args...]
set -uo pipefail

title=$1
shift
log=$(mktemp)
trap 'rm -f "$log"' EXIT

"$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}

if [ "$status" -ne 0 ] && [ -n "${GITHUB_ACTIONS:-}" ]; then
  # Workflow-command data encoding: % as %25, CR as %0D, LF as %0A; colour codes removed.
  message=$(tail -n 50 "$log" | sed -e 's/\x1b\[[0-9;]*[A-Za-z]//g' -e 's/%/%25/g' -e 's/\r/%0D/g' \
    | awk '{ printf "%s%%0A", $0 }')
  echo "::error title=${title} failed (exit ${status})::${message}"
fi
exit "$status"
