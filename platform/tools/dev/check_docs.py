#!/usr/bin/env python3
"""CI doc rule (spec §19): every crate, HAL driver, module, service, and job
must ship a README.md and an examples/ directory. Exits non-zero on violation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # platform/
DOCUMENTED_ROOTS = ["crates", "hal", "modules", "services", "jobs"]


def main() -> int:
    failures: list[str] = []
    for root in DOCUMENTED_ROOTS:
        base = ROOT / root
        if not base.is_dir():
            continue
        for plugin in sorted(p for p in base.iterdir() if p.is_dir()):
            if not (plugin / "README.md").is_file():
                failures.append(f"{root}/{plugin.name}: missing README.md")
            if not (plugin / "examples").is_dir():
                failures.append(f"{root}/{plugin.name}: missing examples/ directory")
    if failures:
        print("Doc rule violations:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Doc rule: OK (all plugins have README.md + examples/).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
