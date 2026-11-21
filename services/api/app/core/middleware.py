from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enforces essential HTTP security headers on all responses."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """Validates Origin / Referer on state-changing requests to defend against CSRF."""

    ALLOWED_ORIGINS = {
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://testserver",
        "http://test",
    }

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            referer = request.headers.get("referer")

            # Check origin if present (allow permitted web origins and Chrome extension origins)
            if origin and not (
                any(origin.startswith(allowed) for allowed in self.ALLOWED_ORIGINS)
                or origin.startswith("chrome-extension://")
            ):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF validation failed: Origin header not permitted."},
                )

            # Check referer if origin not present
            if (
                not origin
                and referer
                and not (
                    any(referer.startswith(allowed) for allowed in self.ALLOWED_ORIGINS)
                    or referer.startswith("chrome-extension://")
                )
            ):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF validation failed: Referer header not permitted."},
                )

        return await call_next(request)
