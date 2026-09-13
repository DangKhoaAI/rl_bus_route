#!/usr/bin/env python
"""Build the native PyO3 kernel, install it as `bus_sim_native`, and record provenance.

Usage:
    python scripts/build_native.py [--release] [--output PATH]

Writes a provenance record (default `reports/rust-migration/native-build.json`)
with toolchain versions, the git revision, the shared-object hash and import
check used by R1+ evidence. Use `--output` for a new kernel revision so the
frozen R4 record stays intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "rust-migration" / "native-build.json"


def _run(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def _workspace_version() -> str:
    payload = tomllib.loads((ROOT / "Cargo.toml").read_text())
    return payload["workspace"]["package"]["version"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true", default=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT,
        help="provenance record to write (default: the frozen rust-migration record)",
    )
    args = parser.parse_args()
    profile = "release" if args.release else "debug"

    build = ["cargo", "build"]
    if args.release:
        build.append("--release")
    build.extend(["-p", "bus-sim-python"])
    print("[native]", " ".join(build), flush=True)
    subprocess.run(build, cwd=ROOT, check=True)

    source = ROOT / "target" / profile / "libbus_sim_native.so"
    target = ROOT / "src" / "bus_sim_native.so"
    shutil.copy2(source, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()

    sys.path.insert(0, str(ROOT / "src"))
    import bus_sim_native

    revision = _run(["git", "rev-parse", "HEAD"])
    dirty = bool(_run(["git", "status", "--porcelain"]))
    payload = {
        "native_build_schema_version": 1,
        "profile": profile,
        "crate": "bus-sim-python",
        "crate_version": _workspace_version(),
        "library": str(target.relative_to(ROOT)),
        "library_sha256": digest,
        "library_bytes": target.stat().st_size,
        "cargo": _run(["cargo", "--version"]),
        "rustc": _run(["rustc", "--version"]),
        "build_command": " ".join(build),
        "install_command": "python scripts/build_native.py",
        "git": {"sha": revision, "dirty": dirty},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "import_ok": hasattr(bus_sim_native, "Kernel"),
        "roles": {
            "r1": "debug kernel bridge for Python-vs-native parity checks",
            "r3": "Gym wrapper is added in Python on top of this module",
        },
    }
    if not payload["import_ok"]:
        raise SystemExit("bus_sim_native.Kernel missing after build")
    report = Path(args.output)
    report = report if report.is_absolute() else ROOT / report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[native] built {digest[:12]} and wrote {report}")


if __name__ == "__main__":
    main()
