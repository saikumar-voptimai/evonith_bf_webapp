#!/usr/bin/env bash
# BF-02 webapp: roll back to an installed release (SV-6.4, #318); release.sh --rollback. Run as root.
#
#   sudo deploy/jetson/rollback.sh            # the previous release (/opt/bf-webapp/previous)
#   sudo deploy/jetson/rollback.sh v1.2.0     # an installed release (/opt/bf-webapp/releases/v1.2.0)
#
# Switches current, restarts bf-webapp.target and runs the smoke test; when the smoke test fails, the release that
# served before the rollback is restored the same way (exit 3). The twin schema is not downgraded (release.sh).
set -euo pipefail
exec "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/release.sh" --rollback "$@"
