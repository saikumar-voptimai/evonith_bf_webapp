"""Phase 12 compatibility shim for the old Streamlit entrypoint.

Use ``streamlit run apps/frontend_streamlit/app.py`` for new startup commands.
This file keeps ``streamlit run src/app.py`` working during the migration.
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path[:0] = [str(_REPO_ROOT)]

runpy.run_path(str(_REPO_ROOT / "apps" / "frontend_streamlit" / "app.py"), run_name="__main__")
