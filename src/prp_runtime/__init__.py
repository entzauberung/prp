"""PRP runtime package root.

This package hosts the reference runtime for the Progressive Reasoning
Protocol. At this version only package identity is defined.
"""

__all__ = ["__version__", "PACKAGE_NAME", "LICENSE_EXPRESSION", "package_info"]

__version__ = "0.0.2"

PACKAGE_NAME = "prp-runtime"

LICENSE_EXPRESSION = "Apache-2.0"


def package_info() -> dict[str, str]:
    """Return the package identity as a plain mapping."""
    return {
        "name": PACKAGE_NAME,
        "version": __version__,
        "license": LICENSE_EXPRESSION,
    }
