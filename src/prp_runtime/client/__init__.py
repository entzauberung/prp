"""Model-free clients for connecting external workers to PRP."""

from prp_runtime.client.bridge import (
    Bridge,
    BridgeClient,
    BridgeError,
    BridgeHTTPError,
    BridgeProtocolError,
    BridgeState,
    BridgeStateError,
    BridgeTransportError,
)
from prp_runtime.client.cli import build_parser, main
from prp_runtime.client.executor import BridgeDispatchError, BridgeDispatchPlan, BridgeExecutor

__all__ = [
    "Bridge",
    "BridgeClient",
    "BridgeError",
    "BridgeHTTPError",
    "BridgeProtocolError",
    "BridgeState",
    "BridgeStateError",
    "BridgeTransportError",
    "BridgeDispatchError",
    "BridgeDispatchPlan",
    "BridgeExecutor",
    "build_parser",
    "main",
]
