"""API v1 router composition."""

from __future__ import annotations

from fastapi import APIRouter

from apps.backend_api.app.api.v1.routes import (
    admin,
    auth,
    blend_optimizer,
    copilot,
    data,
    dashboard,
    datasets,
    feedback,
    furnacemind,
    health,
    jobs,
    material_balance,
    metrics,
    ops,
    plant_admin,
    recommendations,
    status,
    vboard,
)

router = APIRouter()
router.include_router(health.router)
router.include_router(auth.router)
router.include_router(admin.router)
router.include_router(plant_admin.router)
router.include_router(feedback.router)
router.include_router(data.router)
router.include_router(datasets.router)
router.include_router(dashboard.router)
router.include_router(material_balance.router)
router.include_router(recommendations.router)
router.include_router(blend_optimizer.router)
router.include_router(copilot.router)
router.include_router(furnacemind.router)
router.include_router(status.router)
router.include_router(vboard.router)
router.include_router(metrics.router)
router.include_router(jobs.router)
router.include_router(ops.router)
