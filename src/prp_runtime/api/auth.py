"""Single-tenant bearer authentication contracts for Native API dependencies."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from prp_runtime.domain.models import DomainModel
from prp_runtime.domain.values import PrincipalId
from prp_runtime.settings import Settings

__all__ = [
    "AUTHENTICATION_ERROR",
    "PUBLIC_HEALTH_PATH",
    "AuthContext",
    "authenticate",
    "is_public_health_path",
    "require_configured_auth",
    "require_auth",
]

PUBLIC_HEALTH_PATH = "/health"
AUTHENTICATION_ERROR = {
    "code": "authentication_required",
    "message": "a valid bearer token is required",
}

_BEARER = HTTPBearer(auto_error=False)


class AuthContext(DomainModel):
    """The non-secret principal projection passed to business handlers."""

    principal_id: PrincipalId


def authenticate(
    credentials: HTTPAuthorizationCredentials | None,
    settings: Settings,
) -> AuthContext:
    """Validate the configured single service token without retaining its value."""
    configured = settings.service_token
    token = None if configured is None else configured.get_secret_value()
    presented = None if credentials is None else credentials.credentials
    scheme = None if credentials is None else credentials.scheme
    valid = (
        token is not None
        and presented is not None
        and scheme is not None
        and scheme.lower() == "bearer"
        and hmac.compare_digest(presented, token)
    )
    if not valid:
        raise HTTPException(
            status_code=401,
            detail=dict(AUTHENTICATION_ERROR),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return AuthContext(principal_id=settings.service_principal)


async def require_auth(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER)] = None,
) -> AuthContext:
    """FastAPI dependency for all business routes.

    ``/health`` is intentionally public and should not declare this dependency;
    no other path is implicitly exempted by this contract.
    """
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, Settings):
        raise RuntimeError("the application has no authentication settings")
    return authenticate(credentials, settings)


async def require_configured_auth(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER)] = None,
) -> AuthContext:
    """Protect compatibility bindings whenever the service token is configured.

    Older text bindings predate the service-token boundary and remain usable in
    local development with no configured token. A configured deployment fails
    closed for every binding that declares this dependency.
    """
    settings = getattr(request.app.state, "settings", None)
    if not isinstance(settings, Settings):
        raise RuntimeError("the application has no authentication settings")
    if settings.service_token is None:
        return AuthContext(principal_id=settings.service_principal)
    return authenticate(credentials, settings)


def is_public_health_path(path: str) -> bool:
    """Return whether a path is the explicitly anonymous liveness endpoint."""
    return path == PUBLIC_HEALTH_PATH
