"""API v1 router composition."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes import (
    admin,
    auth,
    blend_optimizer,
    data,
    datasets,
    feedback,
    health,
    material_balance,
    recommendations,
)

router = APIRouter()
router.include_router(health.router)
router.include_router(auth.router)
router.include_router(admin.router)
router.include_router(feedback.router)
router.include_router(data.router)
router.include_router(datasets.router)
router.include_router(material_balance.router)
router.include_router(recommendations.router)
router.include_router(blend_optimizer.router)
