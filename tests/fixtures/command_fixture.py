"""Small subprocess fixture for bounded command-runner tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

mode = sys.argv[1]
if mode == "success":
    print(Path.cwd())
elif mode == "failure":
    print("stdout failure")
    print("stderr failure", file=sys.stderr)
    raise SystemExit(3)
elif mode == "flood":
    print("x" * (2 * 1024 * 1024), end="", flush=True)
elif mode == "timeout":
    time.sleep(60)
elif mode == "children":
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=False,
    )
    Path("child.pid").write_text(str(child.pid), encoding="ascii")
    print(child.pid, flush=True)
    time.sleep(60)
elif mode == "env":
    for name in sorted(os.environ):
        print(f"{name}={os.environ[name]}")
elif mode == "sandbox_network":
    routes = Path("/proc/net/route").read_text(encoding="ascii").splitlines()
    if len(routes) > 1:
        print(json.dumps({"network": "present"}))
        raise SystemExit(1)
    print(json.dumps({"network": "isolated"}))
elif mode == "sandbox_sentinel":
    target = Path(sys.argv[2])
    try:
        target.read_text(encoding="ascii")
    except OSError:
        print("sentinel-unmounted")
    else:
        print("sentinel-readable")
        raise SystemExit(1)
elif mode == "sandbox_write":
    target = Path(sys.argv[2])
    try:
        target.write_text("must-not-write", encoding="ascii")
    except OSError:
        print("runtime-read-only")
    else:
        print("runtime-writable")
        raise SystemExit(1)
elif mode == "verify_patch":
    expected = 'def answer():\n    return "patched"\n'
    if Path("src/main.py").read_text(encoding="utf-8") != expected:
        print("patched source does not match", file=sys.stderr)
        raise SystemExit(1)
    print("targeted test passed")
else:
    raise SystemExit(f"unknown mode: {mode}")
