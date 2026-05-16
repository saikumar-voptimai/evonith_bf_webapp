"""Backend-safe service functions for the Evonith BF2 platform.

Modules in this package must not import Streamlit.

Submodules
----------
data_fetch_service   Online/offline fetch, merge, and concat helpers.
ml_service           ML dataset slicing, shift windows, and IST time helpers.
ml_dataset_service   Static ML dataset loading from local CSV or database.
"""
