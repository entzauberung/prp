#!/usr/bin/env python3
"""Generate environment variables from credential file for pytest."""

import sys
from pathlib import Path

# Add parent to path to import credential_loader
sys.path.insert(0, str(Path(__file__).parent))

from credential_loader import load_credentials, PROFILE_CONTRACTS


def main():
    credential_file = Path.home() / "文档/测试key.md"

    if not credential_file.exists():
        print(f"Error: Credential file not found: {credential_file}", file=sys.stderr)
        sys.exit(1)

    try:
        credentials = load_credentials(credential_file)
    except Exception as e:
        print(f"Error loading credentials: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate shell export statements
    print("# Source this file to set up environment variables:")
    print("# source <(python3 external_tests/setup_env.py)")
    print()

    for alias in credentials.aliases:
        env_vars = credentials.profile_env(alias)
        for key, value in env_vars.items():
            print(f'export {key}="{value}"')

    print()
    print("# Environment variables set successfully")


if __name__ == "__main__":
    main()
