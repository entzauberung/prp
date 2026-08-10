"""Module entry point.

Prints package identity. It never starts a server or a background task.
"""

import json

from prp_runtime import package_info


def main() -> int:
    print(json.dumps(package_info(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
