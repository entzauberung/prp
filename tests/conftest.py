"""Shared test constraints.

The default test environment has no real network access. Any attempt to open a
socket fails loudly, and HTTP traffic must be declared through ``respx``.
"""

import socket
from collections.abc import Iterator

import pytest
import respx


class RealNetworkAccessError(RuntimeError):
    """Raised when a test tries to reach the real network."""


@pytest.fixture(autouse=True)
def block_real_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any real outbound socket connection, unless test is marked 'external'."""
    # Skip network blocking for tests marked with @pytest.mark.external
    if "external" in request.keywords:
        return

    def deny(*args: object, **kwargs: object) -> object:
        raise RealNetworkAccessError(
            "real network access is disabled in tests; declare the request with respx"
        )

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)


@pytest.fixture
def mocked_http() -> Iterator[respx.MockRouter]:
    """Provide an HTTP router where an unregistered request is an error."""
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        yield router
