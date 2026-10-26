"""Streamlit-free service layer (programme #262, #271).

Data assembly, caching and model orchestration shared by the FastAPI twin-api, the worker and the Streamlit pages, so
one quantity is computed by one function whichever UI shows it. Nothing under ``services`` imports Streamlit or a
``ui.*`` module; a test enforces this.
"""
