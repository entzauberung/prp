"""Module entry point for the model-free PRP Bridge CLI."""

from prp_runtime.client.cli import main as cli_main


def main() -> int:
    """Delegate module execution to the Bridge CLI."""
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
