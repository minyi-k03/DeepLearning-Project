#!/usr/bin/env python3
"""Single verification entry point for local dev, AI agents, and CI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_AGENTS_MD_BYTES = 32 * 1024


def check_harness() -> None:
    agents_md = ROOT / "AGENTS.md"
    if not agents_md.is_file():
        raise RuntimeError("Missing AGENTS.md at repository root.")
    size = agents_md.stat().st_size
    if size > MAX_AGENTS_MD_BYTES:
        raise RuntimeError(f"AGENTS.md exceeds 32 KiB limit ({size} bytes).")

    rules_dir = ROOT / ".agents" / "rules"
    if rules_dir.is_dir():
        for rule_file in sorted(rules_dir.glob("*.md")):
            text = rule_file.read_text(encoding="utf-8")
            if not text.startswith("---\n") or "trigger:" not in text.split("---", 2)[1]:
                raise RuntimeError(f"Invalid YAML frontmatter/trigger in {rule_file}")

    sync_script = ROOT / "scripts" / "sync_skills.py"
    if sync_script.is_file():
        res = subprocess.run(
            [sys.executable, str(sync_script), "--check"],
            cwd=ROOT,
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError("sync_skills.py --check failed.")


def run_step(label: str, cmd: list[str]) -> None:
    print(f"[verify] {label}: {' '.join(cmd)}")
    env = dict(os.environ)
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_path
    )
    res = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    if res.returncode != 0:
        sys.exit(f"[verify] FAILED at stage '{label}' (exit {res.returncode})")


def main() -> None:
    print("[verify] Checking harness constraints...")
    try:
        check_harness()
    except RuntimeError as exc:
        sys.exit(f"[verify] FAILED at stage 'harness': {exc}")

    run_step("syntax", [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"])
    run_step("pytest", [sys.executable, "-m", "pytest"])
    print("[verify] ALL PASSED")


if __name__ == "__main__":
    main()
