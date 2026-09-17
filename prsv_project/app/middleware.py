from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import Settings


class ApiKeyAndRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Two lightweight, dependency-free protections for /api/* routes:

    1. Optional API key check - if settings.api_key is blank (default), this
       is a no-op, so local/research use is unaffected. Set API_KEY in .env
       to require the X-API-Key header before deploying this beyond your own
       machine (this is the auth story flagged as missing in earlier reviews).

    2. A simple in-memory sliding-window rate limiter, per client IP. This is
       intentionally not a distributed rate limiter (no Redis) - fine for a
       single-machine deployment; resets on restart.

    Both are applied only to /api/ paths - the server-rendered pages (/,
    /analyze, /run/..., etc.) are left alone.
    """

    def __init__(self, app, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings
        self._lock = threading.Lock()
        self._request_log: dict[str, deque] = defaultdict(deque)

    def _client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _check_rate_limit(self, client_ip: str) -> bool:
        now = time.time()
        window = self.settings.rate_limit_window_seconds
        limit = self.settings.rate_limit_requests

        with self._lock:
            log = self._request_log[client_ip]
            while log and now - log[0] > window:
                log.popleft()

            if len(log) >= limit:
                return False

            log.append(now)
            return True

    async def dispatch(self, request: Request, call_next):
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        if self.settings.api_key:
            provided_key = request.headers.get("x-api-key", "")
            if provided_key != self.settings.api_key:
                return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key."})

        client_ip = self._client_ip(request)
        if not self._check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded: max {self.settings.rate_limit_requests} requests per "
                        f"{self.settings.rate_limit_window_seconds}s."
                    )
                },
            )

        return await call_next(request)
