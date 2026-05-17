"""FastAPI application entrypoint."""

import logging
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE any module reads os.environ
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import data, dataset, datasets_v1, health

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
app.include_router(datasets_v1.router)
