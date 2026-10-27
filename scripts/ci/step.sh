#!/usr/bin/env bash
# One CI step (SV-0.11, #280): runs the command with its output in the log as usual. On GitHub Actions:
#   - a failure repeats the last 50 lines as an error annotation, readable on the pull request and through the Checks
#     API without opening the log;
#   - a step slower than STEP_SLOW_S seconds (default 180) adds a notice with its duration, so a slow runner, mirror or
#     download shows which step it delayed.
#
#   bash scripts/ci/step.sh "<title>" <command> [args...]
set -uo pipefail

title=$1
shift
log=$(mktemp)
trap 'rm -f "$log"' EXIT
started=$SECONDS

"$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
elapsed=$((SECONDS - started))

if [ -n "${GITHUB_ACTIONS:-}" ]; then
  if [ "$status" -ne 0 ]; then
    # Workflow-command data encoding: % as %25, CR as %0D, LF as %0A; colour codes removed.
    message=$(tail -n 50 "$log" | sed -e 's/\x1b\[[0-9;]*[A-Za-z]//g' -e 's/%/%25/g' -e 's/\r/%0D/g' \
      | awk '{ printf "%s%%0A", $0 }')
    echo "::error title=${title} failed (exit ${status}) after ${elapsed} s::${message}"
  elif [ "$elapsed" -gt "${STEP_SLOW_S:-180}" ]; then
    echo "::notice title=${title} took ${elapsed} s::Longer than ${STEP_SLOW_S:-180} s."
  fi
fi
exit "$status"
