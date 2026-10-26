"""twin-api: the FastAPI application behind the Svelte frontend (programme #262).

Run from the repository root with ``src/`` on the import path, exactly as the Streamlit app runs:

    python -m uvicorn api.main:app --app-dir src --port 8100

Routers call ``services.*`` only; no route imports Streamlit or a ``ui.*`` module.
"""

__version__ = "0.1.0"
