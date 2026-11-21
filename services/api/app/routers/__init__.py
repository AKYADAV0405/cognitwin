"""API Routers."""

from app.routers.auth import router as auth_router
from app.routers.extension_auth import router as extension_auth_router
from app.routers.health import router as health_router
from app.routers.rbac_smoke import (
    research_router,
    student_router,
)
from app.routers.telemetry import router as telemetry_router

__all__ = [
    "auth_router",
    "extension_auth_router",
    "health_router",
    "student_router",
    "research_router",
    "telemetry_router",
]
