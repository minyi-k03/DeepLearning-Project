# AGENTS.md

## 1. Project Overview
- Read `docs/SPEC.md`, `GLOSSARY.md`, and `docs/adr/` before implementing features.
- ModernBERT-base layer-fusion classification study (Track A); roles: 고용민 (model, fusion, judgment/analysis), 차가원 (infra, extraction/cache, sequential queue runner).

## 2. Source of Truth
- `docs/SPEC.md` defines functional requirements, contracts, and test seams.
- `docs/` planning files are history; `docs/SPEC.md` wins once it exists, and `docs/adr/0001`–`0009` record accepted decisions.
- `GLOSSARY.md` defines canonical terms; `configs/` and `tests/` provide executable verification rules.
- If documents conflict, report the conflict instead of guessing. Never invent values for `docs/SPEC.md` §결정 필요 사항; keep them `null` / `TBD` until confirmed.

## 3. Working Procedure
- At session start, read `docs/progress.md`, `docs/03-plan.md`, and recent git log, then run `python scripts/verify.py`.
- Start a fresh session for an independent task, handoff, or crowded context; update `docs/progress.md` and commit first. Keep small follow-up fixes in the same session.
- Never delete tasks or change acceptance criteria in `docs/progress.md`; ask a human.
- Plan multi-file, seam, contract, or config changes before implementation.
- Preserve existing user changes and report critical ambiguities rather than inventing requirements.

## 4. Collaboration and Architecture Rules
- Work on a feature branch (`feat/<task-id>-<name>`); never commit directly to `main`.
- Shared/high-risk files (`docs/SPEC.md`, `GLOSSARY.md`, `AGENTS.md`, `configs/`, `src/layer_fusion/contracts.py`, `src/layer_fusion/layers.py`, `src/layer_fusion/judgment.py`, `scripts/verify.py`) require teammate review before merging.
- Update contracts in `docs/SPEC.md` / `src/layer_fusion/contracts.py` and tests before changing dependent modules.
- Keep Execution seam (`src/layer_fusion/runner.py`) and Judgment seam (`src/layer_fusion/judgment.py`) CPU-testable with fake providers / synthetic records.
- Do not mix `run_kind="debug"` or `track="reproduction"` records into main judgment/report tables, and do not run multiple `main` timed experiments concurrently on the shared GPU (ADR 0009).

## 5. Skills Usage
- Use relevant installed Skills when the task matches their scope; do not duplicate Skill instructions in this file.
- The source of project Skills is `.agents/skills/`. `.claude/skills/` is a generated copy; never edit it directly—run `python scripts/sync_skills.py` after changing `.agents/skills/`.
- Do not install new Skills or MCP servers without human review.
- Before claiming any task is complete, use `verification-before-completion` (if installed) and run `python scripts/verify.py`.

## 6. Multi-Agent Compatibility
- This file is the single source of shared rules for Antigravity, Claude Code, and Codex.
- `CLAUDE.md` only imports this file (`@AGENTS.md`) and adds Claude-specific rules.
- Do not create `GEMINI.md` or `AGENTS.override.md` without team review.
- Do not assume that one agent's hooks, rules, or permissions apply to other agents.

## 7. Security and Permissions
- Never commit `.env`, credentials, tokens, datasets, or large layer-vector caches.
- Do not overwrite or delete existing experiment caches/results without explicit approval.
- Treat destructive commands and shared-GPU state changes as high-risk operations.

## 8. Verification
- Run `python scripts/verify.py` as the single verification entry point.
- Run relevant `pytest` tests during implementation, and the full `python scripts/verify.py` before completion.
- Check actual seam/CLI behavior, not only static checks.
- Never claim a check succeeded without execution evidence. Report failed or skipped checks separately.

## 9. Completion Report
1. Changes made and their purpose.
2. Important files modified.
3. Verification commands actually executed and results (including skipped/failed checks).
4. Seam or CLI behavior checked.
5. Shared files or contracts changed, and who must be notified.
6. Remaining issues, `TBD` items, and risks.
7. Required approvals or manual actions.
