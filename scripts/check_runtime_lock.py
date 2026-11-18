#!/usr/bin/env python3
"""T-02a: check the final-norm contract per attention/compile setting and time a forward pass.

For every (attn_implementation, torch_compile) combination this records whether the 23rd
hidden state equals the final-normed last-layer output (ADR 0005, SPEC US 24) and, with
--benchmark-steps > 0, the mean forward time on a fixed-length batch. The JSON output is the
evidence for filling `runtime_lock` in configs/week1_env.yaml; values are chosen by the team.

Usage:
  python scripts/check_runtime_lock.py --device cpu --attn sdpa eager
  python scripts/check_runtime_lock.py --device cuda --attn sdpa flash_attention_2 \
      --compile off on --benchmark-steps 20
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import date
from pathlib import Path
from typing import Any

from layer_fusion.extraction import check_final_norm_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attn", nargs="+", default=["sdpa"])
    parser.add_argument("--compile", nargs="+", choices=["off", "on"], default=["off"])
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--benchmark-batch", type=int, default=32)
    parser.add_argument("--benchmark-seq-len", type=int, default=512)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def environment() -> dict[str, Any]:
    import torch
    import transformers

    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def benchmark(
    model_name: str, attn: str, compiled: bool, device: str, steps: int, batch: int, seq_len: int
) -> dict[str, Any]:
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name, attn_implementation=attn).to(device).eval()
    forward = torch.compile(model) if compiled else model
    ids = torch.randint(5, model.config.vocab_size, (batch, seq_len), device=device)
    mask = torch.ones_like(ids)

    def sync() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(2):  # warm-up (includes compilation when enabled)
            forward(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        sync()
        start = time.perf_counter()
        for _ in range(steps):
            forward(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        sync()
    elapsed = time.perf_counter() - start
    return {
        "steps": steps,
        "batch_size": batch,
        "seq_len": seq_len,
        "mean_forward_sec": round(elapsed / steps, 4),
    }


def main() -> None:
    args = parse_args()
    results: list[dict[str, Any]] = []
    for attn in args.attn:
        for compile_flag in args.compile:
            compiled = compile_flag == "on"
            entry: dict[str, Any] = {"attn_implementation": attn, "torch_compile": compiled}
            try:
                report = check_final_norm_contract(
                    args.model_name, attn, args.device, torch_compile=compiled, atol=args.atol
                )
                entry["contract"] = report
                if args.benchmark_steps > 0:
                    entry["benchmark"] = benchmark(
                        args.model_name, attn, compiled, args.device,
                        args.benchmark_steps, args.benchmark_batch, args.benchmark_seq_len,
                    )
            except Exception as exc:  # e.g. flash_attention_2 unavailable on this machine
                entry["error"] = f"{type(exc).__name__}: {exc}"
            results.append(entry)
            status = entry.get("contract", {}).get("passed", entry.get("error"))
            print(f"[check] attn={attn} compile={compile_flag} -> {status}")

    output = Path(args.output or f"docs/measurements/runtime_check_{args.device}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "T-02a",
        "measured_on": date.today().isoformat(),
        "model_name": args.model_name,
        "environment": environment(),
        "results": results,
        "all_contracts_passed": all(r.get("contract", {}).get("passed") for r in results),
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[check] wrote {output}")


if __name__ == "__main__":
    main()
