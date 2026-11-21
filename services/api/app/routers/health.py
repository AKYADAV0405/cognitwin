from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db

router = APIRouter(prefix="/health", tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    database: str
    version: str
    environment: str


@router.get("", response_model=HealthResponse)
async def check_health(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    db_status = "connected"
    overall_status = "healthy"

    try:
        # Verify active database connection
        result = await db.execute(text("SELECT 1"))
        result.scalar()
    except Exception as exc:
        db_status = f"disconnected: {exc!s}"
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        database=db_status,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
    )
