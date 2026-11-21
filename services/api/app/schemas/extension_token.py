import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ExtensionTokenCreate(BaseModel):
    name: str = Field(default="Chrome Extension", max_length=100)


class ExtensionTokenResponse(BaseModel):
    token: str = Field(description="Raw scoped Bearer token. Stored hashed on server.")
    name: str
    created_at: datetime


class ExtensionTokenInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    last_used_at: datetime | None
    created_at: datetime
    is_active: bool
