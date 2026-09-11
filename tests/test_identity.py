from prp import __version__


def test_package_identity() -> None:
    assert __version__ == "0.0.1"
