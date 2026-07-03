"""API v1 router composition."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes import data, datasets, health

router = APIRouter()
router.include_router(health.router)
router.include_router(data.router)
router.include_router(datasets.router)
