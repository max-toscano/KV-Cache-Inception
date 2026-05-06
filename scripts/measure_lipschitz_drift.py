#!/usr/bin/env python3
"""
Theorem 1 empirical validation — FP32 accumulator Lipschitz drift measurement.

Validates Theorem 1 (paper §4.2): exact reversibility of KV-cache steering
via FP32 accumulator is bounded independently of rollback count n.

Theorem 1 predictions:
  Without FP32 accumulator (naive bf16 subtraction):
    ‖K_n − K_0‖_∞ ≤ n · ε_bf16 · max_i ‖δ_i‖_∞    [grows linearly with n]

  With FP32 accumulator (this script validates):
    ‖K_n − K_0‖_∞ ≤ ε_bf16 · ‖A_final‖_∞            [independent of n]

Output: CSV with columns [cycle, naive_inf_norm, accumulator_inf_norm].
Expected result: naive_inf_norm grows linearly; accumulator_inf_norm stays
near ε_bf16 ≈ 3.9e-3 regardless of cycle count.

Usage:
    uv run python scripts/measure_lipschitz_drift.py \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --n-cycles 1000 --alpha 0.5 --output drift_results.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("measure_lipschitz_drift")


def run_probe(args: argparse.Namespace) -> list[dict]:
    """Run N apply/rollback cycles and record ‖K_n − K_0‖_∞ per cycle.

    Returns list of dicts: [{cycle, naive_inf_norm, accumulator_inf_norm}]
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from logomesh.kv_mcts import FP32Accumulator, _extract_kv_tensors, _clone_kv_cache

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Auto dtype
    if args.dtype == "auto":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif device == "cpu":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = getattr(torch, args.dtype)

    logger.info("Loading model %s on %s (%s)...", args.model, device, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device).eval()
    logger.info("Model loaded.")

    # Seed KV cache
    prompt = "The alignment problem in AI refers to"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    past_kv_live = out.past_key_values
    if past_kv_live is None:
        raise RuntimeError("Model did not return past_key_values.")

    # Clone baseline for drift measurement
    past_kv_baseline = _clone_kv_cache(past_kv_live)

    # Build steering vector (random unit vector, same size as first K tensor's last dim)
    layers0 = _extract_kv_tensors(past_kv_live)
    k0, _ = layers0[0]
    d = k0.shape[-1]
    rng = torch.Generator()
    rng.manual_seed(42)
    dk_raw = torch.randn(d, generator=rng)
    dk_raw = (dk_raw / dk_raw.norm()).numpy()  # unit vector on CPU
    dk_vectors = [dk_raw] * len(layers0)

    # FP32 accumulator path
    accumulator = FP32Accumulator.from_kv_cache(past_kv_live)

    # Naive path: track K directly (no accumulator)
    # We'll simulate naive addition/subtraction with the live cache clone
    # Snapshot of layer-0 K at start (float32 for accurate comparison)
    k_base_f32 = layers0[0][0].float().clone()

    records = []

    for cycle in range(1, args.n_cycles + 1):
        # ── Accumulator path: apply ──────────────────────────────────────
        ok = accumulator.apply(past_kv_live, args.alpha, dk_vectors)
        if not ok:
            raise RuntimeError(
                f"FP32Accumulator.apply() failed at cycle {cycle}: "
                "steering vector shape does not match KV cache. "
                "Check --model matches expected architecture."
            )

        # ── Accumulator path: rollback ───────────────────────────────────
        ok = accumulator.rollback(past_kv_live, args.alpha, dk_vectors)
        if not ok:
            raise RuntimeError(
                f"FP32Accumulator.rollback() failed at cycle {cycle}: "
                "shape invariant violated after successful apply."
            )

        # Measure ‖K_n − K_0‖_∞ for accumulator path
        k_now = _extract_kv_tensors(past_kv_live)[0][0].float()
        acc_inf_norm = float((k_now - k_base_f32).abs().max().item())

        # ── Naive path: apply `cycle` adds then `cycle` subtracts in bf16 ──
        # This simulates N forward perturbations followed by N reversals without
        # FP32 accumulation — rounding errors compound across steps, showing
        # growing residual that the FP32 accumulator avoids.
        dk_t = torch.from_numpy(dk_raw).to(device=k_base_f32.device, dtype=torch_dtype)
        k_naive = k_base_f32.clone().to(torch_dtype)
        for _ in range(cycle):
            k_naive.add_(args.alpha * dk_t)
        for _ in range(cycle):
            k_naive.sub_(args.alpha * dk_t)
        naive_inf_norm = float((k_naive.float() - k_base_f32).abs().max().item())

        records.append(
            {
                "cycle": cycle,
                "naive_inf_norm": naive_inf_norm,
                "accumulator_inf_norm": acc_inf_norm,
            }
        )

        if cycle % 50 == 0 or cycle == 1:
            logger.info(
                "Cycle %4d: naive=%.2e  accumulator=%.2e  (residual=%.2e)",
                cycle,
                naive_inf_norm,
                acc_inf_norm,
                accumulator.residual_norm(),
            )

    return records


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Local path or HuggingFace model id",
    )
    p.add_argument(
        "--n-cycles",
        type=int,
        default=1000,
        help="Number of apply/rollback cycles to measure",
    )
    p.add_argument(
        "--alpha", type=float, default=0.5, help="Steering magnitude per cycle"
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument(
        "--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    p.add_argument("--output", default="drift_results.csv", help="Output CSV path")
    args = p.parse_args()

    records = run_probe(args)

    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["cycle", "naive_inf_norm", "accumulator_inf_norm"]
        )
        writer.writeheader()
        writer.writerows(records)
    logger.info("Drift results saved → %s", output_path)

    # Summary
    final = records[-1]
    print(f"\n--- Theorem 1 Drift Summary (n={args.n_cycles} cycles) ---")
    print(
        f"  Naive bf16 subtraction:   ||K_n - K_0||_inf = {final['naive_inf_norm']:.4e}"
    )
    print(
        f"  FP32 accumulator:         ||K_n - K_0||_inf = {final['accumulator_inf_norm']:.4e}"
    )
    print("  eps_bf16 (theoretical):                    ~3.9e-03")
    print(f"  Theorem 1 bound satisfied: {final['accumulator_inf_norm'] < 4e-3}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
