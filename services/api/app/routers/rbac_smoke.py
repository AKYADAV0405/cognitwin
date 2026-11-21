from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.dependencies import require_role
from app.models.user import User

student_router = APIRouter(prefix="/student", tags=["Student"])
research_router = APIRouter(prefix="/research", tags=["Researcher"])


class PingResponse(BaseModel):
    message: str
    user_email: str
    role: str


@student_router.get("/ping", response_model=PingResponse)
async def student_ping(
    current_user: User = Depends(require_role("student")),
) -> PingResponse:
    """Accessible by both student and researcher roles."""
    return PingResponse(
        message="Student workspace access granted.",
        user_email=current_user.email,
        role=current_user.role,
    )


@research_router.get("/ping", response_model=PingResponse)
async def research_ping(
    current_user: User = Depends(require_role("researcher")),
) -> PingResponse:
    """Strictly accessible only by the researcher role; returns 403 for students."""
    return PingResponse(
        message="Researcher workspace access granted.",
        user_email=current_user.email,
        role=current_user.role,
    )
