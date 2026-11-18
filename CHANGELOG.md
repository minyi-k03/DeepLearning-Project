# Changelog

## Unreleased
- T-02b: add `scripts/measure_imdb_lengths.py`, record IMDB token-length measurement (`docs/measurements/imdb_length_check.json`) and fill `imdb_length_check` in `configs/week1_env.yaml` (max 3,270 tokens, no Lfull truncation).
- T-02a: pin `transformers==5.23.0`, implement `check_final_norm_contract` / `verify_final_norm_contract` and `masked_token_mean` in `extraction.py`, add `scripts/check_runtime_lock.py` (contract check + optional forward timing per attention/compile setting), record CPU result in `docs/measurements/runtime_check_cpu.json`, fill `transformers_version` and `last_vector_is_final_norm` in `configs/week1_env.yaml` (attention/compile pending GPU measurement).
- T-01a: pin Python 3.11 (`.python-version`), add cross-platform `requirements.lock` (uv), and document the install/verify procedure in `README.md`.
- Review follow-up: point references to `docs/SPEC.md`, fix ruff findings and add a `ruff check` stage to `scripts/verify.py`, move the workflow guide to `docs/03-capstone-project.md`, drop the unsupported `Read(!.env.example)` permission rule.
- Initial repository structure, configuration templates, module skeletons, test seams, and harness files created from `SPEC.md` and `docs/adr/0001`–`0009`.
