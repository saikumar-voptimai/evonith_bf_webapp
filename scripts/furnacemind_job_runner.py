"""Stable executable wrapper for the headless scheduled-job runner.

Deployment invokes this file with the application virtual-environment Python.
It adds the repository ``src`` directory to the module path and delegates all
behavior to ``utils.scheduled_tasks.job_runner``. Keeping this wrapper thin
avoids duplicating runner or persistence logic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# A systemd worker must consume only its root-controlled EnvironmentFile. This
# is set before importing application modules so ``utils.settings`` cannot read
# a developer ``.env`` file from the working directory.
os.environ["FURNACEMIND_DISABLE_DOTENV"] = "true"

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from utils.scheduled_tasks.job_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
