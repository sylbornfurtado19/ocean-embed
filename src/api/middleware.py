"""Production-grade security and reliability middleware for OceanEmbed API."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import datetime
import logging
import time
from typing import AsyncGenerator, Callable
import uuid

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import settings

logger = logging.getLogger("oceanembed.api.middleware")


class RequestIdAndSecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware attaching request IDs, logging request lifecycle, and setting security headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()

        # Extract or generate unique Request ID
        req_id = request.headers.get("X-Request-ID")
        if not req_id or len(req_id) > 64 or not all(c.isalnum() or c in "-_" for c in req_id):
            req_id = str(uuid.uuid4())

        request.state.request_id = req_id

        # Check body size from Content-Length header early
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.max_request_body_bytes:
                    logger.warning(
                        "[%s] Request body size %s exceeds limit %s bytes",
                        req_id,
                        content_length,
                        settings.max_request_body_bytes,
                    )
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "detail": f"Request body exceeds maximum size of {settings.max_request_body_bytes} bytes.",
                            "error_code": "PAYLOAD_TOO_LARGE",
                            "request_id": req_id,
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        },
                        headers={"X-Request-ID": req_id},
                    )
            except ValueError:
                pass

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error("[%s] Unhandled exception during request dispatch: %s", req_id, exc, exc_info=True)
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": "Internal server error occurred during request processing.",
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "request_id": req_id,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                },
            )

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Inject Security Headers & Request ID
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if "Cache-Control" not in response.headers:
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"

        logger.info(
            "[%s] %s %s -> %d (%.2fms)",
            req_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


class InMemoryRateLimiterMiddleware(BaseHTTPMiddleware):
    """Process-local sliding window rate limiter for expensive prediction requests."""

    def __init__(self, app, max_requests_per_minute: int | None = None) -> None:
        super().__init__(app)
        self._configured_rpm = max_requests_per_minute
        self._client_records: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Rate limit only expensive prediction endpoints
        if request.method == "POST" and request.url.path.rstrip("/").endswith("/predict"):
            client_ip = request.client.host if request.client else "unknown_client"
            now = time.time()
            cutoff = now - 60.0
            max_rpm = self._configured_rpm or settings.rate_limit_per_minute

            async with self._lock:
                history = self._client_records.setdefault(client_ip, [])
                # Prune entries older than 60 seconds
                history = [t for t in history if t > cutoff]
                self._client_records[client_ip] = history

                if len(history) >= max_rpm:
                    req_id = getattr(request.state, "request_id", "unknown")
                    logger.warning("[%s] Rate limit exceeded for client %s (%d rpm)", req_id, client_ip, len(history))
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={
                            "detail": f"Rate limit of {max_rpm} requests per minute exceeded. Please retry later.",
                            "error_code": "RATE_LIMIT_EXCEEDED",
                            "request_id": req_id,
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        },
                        headers={
                            "Retry-After": "60",
                            "X-Request-ID": req_id,
                        },
                    )

                history.append(now)


        return await call_next(request)


class InferenceConcurrencyLimiter:
    """Bounded concurrency manager preventing CPU/memory exhaustion on ML forward passes."""

    def __init__(self, max_concurrent: int | None = None, acquire_timeout: float = 2.0) -> None:
        self.max_concurrent = max_concurrent or settings.max_concurrent_inference
        self.acquire_timeout = acquire_timeout
        self._semaphore: asyncio.Semaphore | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    @asynccontextmanager
    async def slot(self, request_id: str = "") -> AsyncGenerator[None, None]:
        """Acquire an inference execution slot with bounded wait time."""
        sem = self._get_semaphore()
        acquired = False
        try:
            try:
                acquired = await asyncio.wait_for(sem.acquire(), timeout=self.acquire_timeout)
            except asyncio.TimeoutError:
                acquired = False

            if not acquired:
                logger.warning("[%s] Inference concurrency limit (%d) reached", request_id, self.max_concurrent)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Server is currently handling maximum concurrent model inferences. Please retry momentarily.",
                )
            yield
        finally:
            if acquired:
                sem.release()


# Global concurrency limiter instance
inference_concurrency_limiter = InferenceConcurrencyLimiter()
