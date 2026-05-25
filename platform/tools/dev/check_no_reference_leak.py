#!/usr/bin/env python3
"""CI guard: no production source under ``platform/`` may import from or
path into the frozen ``reference/`` tree (rev-2 plan, README §22).

``reference/`` is the prior VGR codebase, retained read-only during the
rebuild and deleted on release. If any production source links back into
it, that deletion silently breaks. This script catches the leak at PR time.

Allowed: documentation files (``*.md``) may mention ``reference/`` -- the
per-module READMEs name their porting origin (e.g. "Migrates from
reference: ...") and that's the docs that survive a release.

Banned everywhere (incl. .md): a Python ``import reference`` / ``from reference``.
Banned in code: any literal ``reference/`` path (used in imports, file
loaders, build scripts, configs).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[2]  # repo/platform/
SKIP_DIRS = {
    ".git", "__pycache__", "target", "node_modules",
    ".pytest_cache", "dist", "build",
}
SKIP_FILES = {
    Path("tools/dev/check_no_reference_leak.py"),  # this file
}
# Code files where a literal "reference/" is a real leak.
CODE_EXTS = {
    ".py", ".rs", ".toml", ".yaml", ".yml", ".json",
    ".ts", ".tsx", ".js", ".html", ".css", ".sh", ".proto",
}
# Docs/text where path mentions are OK (Python imports are still banned).
DOC_EXTS = {".md"}

PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9_])reference/")
# Python import context: `from reference` or `import reference` followed by
# whitespace, a dot, or end-of-line. README sentences like "from reference:"
# (with a punctuation tail) are not imports and do not match.
IMPORT_PATTERN = re.compile(r"\b(from|import)\s+reference(?=[\s\.]|$)", re.MULTILINE)


def _should_skip(path: Path) -> bool:
    try:
        rel = path.relative_to(PLATFORM)
    except ValueError:
        return True
    if rel in SKIP_FILES:
        return True
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    # The vendored algorithms/ package is a self-contained copy of
    # robot-algorithms; it has no relation to reference/ and its own README
    # references its upstream history.
    if "modules/motion-core/algorithms" in str(rel):
        return True
    return False


def _record(violations: list[str], path: Path, text: str, pat: re.Pattern[str]) -> None:
    rel = path.relative_to(PLATFORM)
    for m in pat.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        snippet = text.splitlines()[line - 1].strip()[:120]
        violations.append(f"{rel}:{line}: {snippet}")


def main() -> int:
    violations: list[str] = []
    for path in PLATFORM.rglob("*"):
        if not path.is_file() or _should_skip(path):
            continue
        ext = path.suffix.lower()
        if ext not in CODE_EXTS and ext not in DOC_EXTS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Python-style imports are banned in every file type.
        _record(violations, path, text, IMPORT_PATTERN)
        # Path-style "reference/" only banned in code, not docs.
        if ext in CODE_EXTS:
            _record(violations, path, text, PATH_PATTERN)

    if violations:
        print(f"Found {len(violations)} reference/ leak(s) in platform/ code:")
        for v in violations:
            print(f"  - {v}")
        print("\nNothing under platform/ (except *.md docs) may import from or path into reference/.")
        return 1
    print("No reference/ leaks under platform/ code. OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
