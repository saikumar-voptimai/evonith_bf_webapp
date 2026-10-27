"""Start the Streamlit app (``src/app.py``) from any directory, on any platform.

    python run_streamlit.py                                   # http://localhost:8501
    python run_streamlit.py --server.baseUrlPath legacy       # under /legacy, as behind Caddy (SV-0.10, #279)

Arguments are passed to ``streamlit run``. The working directory becomes the repository root, as the pages expect.
On Windows, torch is imported before Streamlit: the opposite order fails with WinError 1114 (DLL initialisation).
Use this script rather than ``streamlit run`` directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "src" / "app.py"


def main(argv: list[str] | None = None) -> int:
    if os.name == "nt":
        import torch  # noqa: F401  (must precede Streamlit on Windows)
    try:
        from streamlit.web.cli import main as streamlit_main
    except ImportError:  # older Streamlit releases
        from streamlit.web import cli as stcli

        streamlit_main = stcli.main
    os.chdir(ROOT)
    sys.argv = ["streamlit", "run", str(APP), *(sys.argv[1:] if argv is None else argv)]
    return streamlit_main()


if __name__ == "__main__":
    raise SystemExit(main())
