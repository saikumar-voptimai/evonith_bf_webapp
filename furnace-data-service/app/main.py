"""FastAPI application entrypoint."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE any module reads os.environ
_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SERVICE_ROOT.parent

load_dotenv(_SERVICE_ROOT / ".env")
os.environ.setdefault("FURNACE_CONFIG_DIR", str(_REPO_ROOT / "src" / "config"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import data, dataset, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Furnace Data Service",
    description="REST API for fetching and processing Blast Furnace datasets",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(data.router)
app.include_router(dataset.router)
