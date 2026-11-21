from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = "Cognitwin API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/cognitwin"

    # JWT & Auth
    JWT_SECRET_KEY: str = "dev-secret-key-change-in-production-min-32-bytes"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Cookie security (False for HTTP local development, True for HTTPS production)
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"

    # Rate Limiting toggle
    RATE_LIMIT_ENABLED: bool = True

    # Google OAuth 2.0 / OIDC Settings
    GOOGLE_CLIENT_ID: str = "mock-google-client-id.apps.googleusercontent.com"
    GOOGLE_CLIENT_SECRET: str = "mock-google-client-secret"
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/google/callback"
    FRONTEND_URL: str = "http://localhost:5173"


settings = Settings()
