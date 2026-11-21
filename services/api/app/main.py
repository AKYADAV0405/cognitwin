from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from sqlalchemy import inspect, text

import app.models  # noqa: F401
from app.core.config import settings
from app.core.database import Base, engine
from app.core.limiter import limiter
from app.core.middleware import CSRFProtectionMiddleware, SecurityHeadersMiddleware
from app.routers import (
    auth_router,
    extension_auth_router,
    health_router,
    research_router,
    student_router,
    telemetry_router,
)


def _sync_sqlite_schema(sync_conn: Any) -> None:
    """Auto-migrates newly added columns in local SQLite development mode."""
    inspector = inspect(sync_conn)
    for table_name, table in Base.metadata.tables.items():
        if inspector.has_table(table_name):
            existing_cols = {col["name"] for col in inspector.get_columns(table_name)}
            for col in table.columns:
                if col.name not in existing_cols:
                    col_type = col.type.compile(sync_conn.dialect)
                    default_clause = ""
                    if not col.nullable:
                        type_str = str(col_type).upper()
                        if any(k in type_str for k in ("INT", "FLOAT", "NUMERIC", "BOOL")):
                            default_clause = " DEFAULT 0"
                        else:
                            default_clause = " DEFAULT ''"
                    sync_conn.execute(
                        text(
                            f"ALTER TABLE {table_name} ADD COLUMN {col.name} "
                            f"{col_type}{default_clause}"
                        )
                    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize database tables for local SQLite development if needed."""
    if "sqlite" in settings.DATABASE_URL:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                await conn.run_sync(_sync_sqlite_schema)
        except Exception:
            pass
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Cognitwin Core Telemetry, Digital Twin & Knowledge Tracing API",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    redoc_url=f"{settings.API_V1_PREFIX}/redoc",
    lifespan=lifespan,
)

# Connect slowapi rate limiter state
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Returns a structured 429 response with Retry-After header."""
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please slow down."},
        headers={"Retry-After": "60"},
    )


# Security and CSRF Middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(CSRFProtectionMiddleware)

# CORS configuration for local development with HTTP-only credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"^chrome-extension://.*$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 routers
app.include_router(health_router, prefix=settings.API_V1_PREFIX)
app.include_router(auth_router, prefix=settings.API_V1_PREFIX)
app.include_router(extension_auth_router, prefix=settings.API_V1_PREFIX)
app.include_router(telemetry_router, prefix=settings.API_V1_PREFIX)
app.include_router(student_router, prefix=settings.API_V1_PREFIX)
app.include_router(research_router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs": f"{settings.API_V1_PREFIX}/docs",
    }
