"""PRP runtime package root.

This package hosts the reference runtime for the Progressive Reasoning
Protocol: local-first controller, policy, tools, evidence and recovery.
"""

__all__ = ["__version__", "PACKAGE_NAME", "LICENSE_EXPRESSION", "package_info"]

__version__ = "0.0.4"

PACKAGE_NAME = "prp-runtime"

LICENSE_EXPRESSION = "Apache-2.0"


def package_info() -> dict[str, str]:
    """Return the package identity as a plain mapping."""
    return {
        "name": PACKAGE_NAME,
        "version": __version__,
        "license": LICENSE_EXPRESSION,
    }
