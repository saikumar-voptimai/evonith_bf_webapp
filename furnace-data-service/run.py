"""Uvicorn entrypoint for the ML Dataset API."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

_SIDECAR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SIDECAR_DIR.parent

load_dotenv(_SIDECAR_DIR / ".env")

# Use the monorepo's main YAML config until the UI/backend split.  The sidecar
# no longer ships its own `furnace-data-service/config/setting_ds_dv.yml`; this
# env var redirects `furnace_data.config.load_config(...)` to the canonical one.
os.environ.setdefault("FURNACE_CONFIG_DIR", str(_REPO_ROOT / "src" / "config"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "apps.backend_api.app.main:app",
        host=settings.host,
        port=settings.port,   # default 8080, override via PORT env var
        reload=True,
    )
