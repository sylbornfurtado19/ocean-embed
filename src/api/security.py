"""Security dependencies for OceanEmbed API authentication."""

from __future__ import annotations

import logging
import secrets
from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from src.config import settings

logger = logging.getLogger("oceanembed.api.security")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> str | None:
    """Verify incoming X-API-Key header when API_KEY is configured.

    - When settings.api_key is empty/unset, authentication is disabled (local dev/demo).
    - When settings.api_key is set, protected routes require a matching X-API-Key.
    """
    configured_key = settings.api_key
    if not configured_key:
        # Authentication disabled in current environment
        return None

    if not api_key:
        logger.warning("Unauthenticated request to %s: missing X-API-Key header", request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required API key in X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, configured_key):
        logger.warning("Unauthorized request to %s: invalid API key provided", request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key provided.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key
