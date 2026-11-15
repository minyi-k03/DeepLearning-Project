#!/usr/bin/env python3
"""Sync .agents/skills/ (source) to .claude/skills/ (copy).

Usage:
  python scripts/sync_skills.py           # Sync (aborts if destination has extra files)
  python scripts/sync_skills.py --check   # Exit 1 on mismatch (for CI / verify)
  python scripts/sync_skills.py --force   # Overwrite destination including extra files
"""

from __future__ import annotations

import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / ".agents" / "skills"
DST = ROOT / ".claude" / "skills"


def differs(a: Path, b: Path) -> bool:
    if not b.is_dir():
        return True
    cmp = filecmp.dircmp(a, b)
    if cmp.left_only or cmp.right_only or cmp.funny_files:
        return True
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    if mismatch or errors:
        return True
    return any(differs(a / d, b / d) for d in cmp.common_dirs)


def copy_only_items(src: Path, dst: Path) -> list[Path]:
    if not dst.is_dir():
        return []
    items: list[Path] = []
    for path in sorted(dst.rglob("*")):
        rel = path.relative_to(dst)
        if not (src / rel).exists() and (src / rel.parent).exists():
            items.append(rel)
    return items


def main() -> None:
    if not SRC.is_dir():
        sys.exit(f"Source directory missing: {SRC}")
    if "--check" in sys.argv:
        if differs(SRC, DST):
            sys.exit("Skills out of sync: run `python scripts/sync_skills.py` and commit.")
        print("Skills sync OK")
        return

    extras = copy_only_items(SRC, DST)
    if extras and "--force" not in sys.argv:
        print("Extra items found only in .claude/skills/:")
        for rel in extras[:30]:
            print(f"  - {rel}")
        sys.exit("Move needed skills to .agents/skills/ or re-run with --force.")

    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    print(f"Synced {SRC.relative_to(ROOT)} -> {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
