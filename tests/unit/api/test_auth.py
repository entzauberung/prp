"""Targeted tests for the single-tenant bearer authentication contract."""

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import SecretStr

from prp_runtime.api.auth import (
    PUBLIC_HEALTH_PATH,
    authenticate,
    is_public_health_path,
)
from prp_runtime.settings import Settings


def credentials(token: str, scheme: str = "Bearer") -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme=scheme, credentials=token)


def test_valid_bearer_returns_only_stable_principal() -> None:
    settings = Settings(
        service_token=SecretStr("server-secret"),
        service_principal="prn_operator",
    )

    context = authenticate(credentials("server-secret"), settings)

    assert context.principal_id == "prn_operator"
    assert "server-secret" not in repr(settings)
    assert "server-secret" not in settings.model_dump_json()
    assert "service_token" not in context.model_dump()


def test_missing_wrong_and_non_bearer_tokens_are_structured_401() -> None:
    settings = Settings(service_token=SecretStr("server-secret"))

    for presented in (None, credentials("wrong"), credentials("server-secret", "Basic")):
        try:
            authenticate(presented, settings)
        except HTTPException as error:
            assert error.status_code == 401
            assert error.headers == {"WWW-Authenticate": "Bearer"}
            assert error.detail["code"] == "authentication_required"
        else:
            raise AssertionError("invalid credentials were accepted")


def test_auth_without_server_token_fails_closed() -> None:
    try:
        authenticate(credentials("anything"), Settings())
    except HTTPException as error:
        assert error.status_code == 401
    else:
        raise AssertionError("authentication succeeded without server configuration")


def test_only_health_is_anonymous_by_contract() -> None:
    assert is_public_health_path(PUBLIC_HEALTH_PATH)
    assert not is_public_health_path("/ready")
    assert not is_public_health_path("/v1/runs")
