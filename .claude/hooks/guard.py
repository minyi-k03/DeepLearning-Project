#!/usr/bin/env python3
"""Claude Code PreToolUse(Bash|PowerShell) guard.

Blocks direct commit/push/merge/rebase on main/master, force push,
and shell access to .env files (excluding .env.example).
"""

import json
import re
import subprocess
import sys

try:
    data = json.load(sys.stdin)
except ValueError:
    sys.exit(0)

cmd = (data.get("tool_input") or {}).get("command", "") or ""

GIT = r"\bgit(?:\s+-[cC]\s+\S+|\s+--?[\w-]+(?:=\S+)?)*\s+"
BRANCH = r"['\"]?(main|master)(?=$|[\s;&|'\")`])"
ARGS = r"[^;&|\n]*"


def block(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(2)


def current_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=data.get("cwd") or None,
            timeout=5,
            check=False,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


if re.search(GIT + r"push\b" + ARGS + r"(\s--force\S*|\s-[a-zA-Z]*f[a-zA-Z]*\b|\s\+\S)", cmd):
    block("강제 push는 금지입니다. 필요하면 팀원에게 요청하세요.")

if re.search(GIT + r"push\b" + ARGS + r"(?:[\s:+]|refs/heads/)" + BRANCH, cmd):
    block("main으로 직접 push 금지: 기능 브랜치를 push하고 PR로 병합하세요.")

FORCE_MOVE = (
    r"(?:branch(?=" + ARGS + r"\s(?:-[a-zA-Z]*f[a-zA-Z]*\b|--force\b))(?:\s+-\S+)*"
    r"|checkout(?:\s+-\S+)*?\s+-B|switch(?:\s+-\S+)*?\s+-C)\s+"
)
if re.search(GIT + FORCE_MOVE + BRANCH, cmd):
    block("main 브랜치를 강제로 옮기는 명령은 금지입니다.")

WRITE = GIT + r"(commit|push|merge|rebase|cherry-pick)\b"
if re.search(WRITE, cmd) and current_branch() in ("main", "master"):
    block("main 브랜치에서 commit/push/merge 금지: feat/<task-id>-<name> 브랜치로 전환하세요.")

NOT_SWITCH = r"(?!" + ARGS + r"\s(?:-[bBcC]\b|--create\b|--force-create\b|--orphan\b|--(?=\s|$)))"
to_main = re.search(GIT + r"(checkout|switch)\b" + NOT_SWITCH + ARGS + r"\s" + BRANCH, cmd)
if to_main and re.search(WRITE, cmd[to_main.end() :]):
    block("main으로 전환한 뒤 commit/push/merge하는 명령은 금지입니다.")

scan = re.sub(r"""\s-m\s*(?:"[^"]*"|'[^']*')""", " ", cmd)
if re.search(
    r"(?<![\w.-])(?:[\w-]+\.env|\.env(?:\.(?!example(?=$|[\s'\";|&>)},`]))[\w.-]+)?)(?=$|[\s'\";|&>)},`])",
    scan,
) or re.search(r"(?<![\w.-])\.env?[*?\[]", scan):
    block(".env 파일 접근 금지: 변수 이름은 .env.example을 참고하세요.")

sys.exit(0)
