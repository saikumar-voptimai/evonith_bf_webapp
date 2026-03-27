import sys

# Import torch first to avoid WinError 1114 triggered by import order on Windows
import torch  # noqa: F401

# Start Streamlit after torch import
try:
    from streamlit.web import cli as stcli
    main = stcli.main
except Exception:
    # fallback for some Streamlit versions
    from streamlit.web.cli import main  # type: ignore

sys.argv = ["streamlit", "run", r".\src\app.py"]
raise SystemExit(main())