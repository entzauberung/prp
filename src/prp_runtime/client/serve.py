"""Optional loopback ASGI serve dispatch over the existing app factory."""

from __future__ import annotations

from collections.abc import Mapping

from prp_runtime.app import create_app
from prp_runtime.providers.base import ProviderAdapter
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore

__all__ = ["MAX_SERVE_PORT", "MIN_SERVE_PORT", "build_serve_app", "serve_app"]

MIN_SERVE_PORT = 1
MAX_SERVE_PORT = 65_535
_DEFAULT_HOST = "127.0.0.1"


def build_serve_app(
    settings: Settings | None = None,
    *,
    adapters: Mapping[str, ProviderAdapter] | None = None,
    store: SqliteStore | None = None,
):
    """Build the existing ASGI app; tests must not start a listener."""
    return create_app(settings or Settings.from_env(), adapters=adapters, store=store)


def serve_app(
    *,
    host: str = _DEFAULT_HOST,
    port: int = 8000,
    settings: Settings | None = None,
    adapters: Mapping[str, ProviderAdapter] | None = None,
    store: SqliteStore | None = None,
) -> None:
    """Serve ``create_app`` through the installed ASGI runner."""
    if port < MIN_SERVE_PORT or port > MAX_SERVE_PORT:
        raise ValueError("serve port must be between 1 and 65535")
    import uvicorn

    uvicorn.run(
        build_serve_app(settings, adapters=adapters, store=store),
        host=host,
        port=port,
    )
