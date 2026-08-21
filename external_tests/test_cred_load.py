#!/usr/bin/env python3
"""Debug credential loading."""

from external_tests.credential_loader import parse_credentials_text
from pathlib import Path

cred_file = Path("/home/bruce/文档/测试key.md")
text = cred_file.read_text()

print("=== Raw content ===")
for i, line in enumerate(text.splitlines()[:50], 1):
    print(f"{i:3}: {line}")

print("\n=== Attempting parse ===")
try:
    creds = parse_credentials_text(text)
    print(f"Success! Loaded {len(creds.aliases)} profiles")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
