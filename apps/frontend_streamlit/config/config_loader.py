"""Frontend YAML configuration loader."""

from __future__ import annotations

def load_config(*args, **kwargs):
    """Load shared configuration only when a direct-mode caller needs it."""
    from furnace_data.config import load_config as _load_config

    return _load_config(*args, **kwargs)
