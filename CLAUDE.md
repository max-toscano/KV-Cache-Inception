# LogoMesh — Research Session Guide

> **Read this file at the start of every session.**
> Then check `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` for the current paper state
> and `docs/reviews/phase-a-gaps-2026-04-11.md` for the verified gap analysis.

---

## Project

**KV-Cache Inception: Reversible Monte Carlo Tree Search in Latent Space for Detecting and
Stress-Testing Alignment Faking in Large Language Models**

NeurIPS 2026 Datasets & Benchmarks track. Submission deadline: **May 6, 2026**.

Academic advisor: Prof. Tianyu Shi (McGill). Contributors: Josh, Max, Bakul, Asuka.

---

## Three Paper Contributions

1. **Multi-Dimensional Micro-Telemetry Matrix** — H-Neuron activation (bottom-up) fused with
   RepE linear probes (top-down) into a per-layer, per-step matrix `T_t ∈ ℝ^{2×L}`. Defines
   four diagnostic states: Creative Synthesis, Contextual Fabrication, Alignment Faking,
   Orthogonal Escape.

2. **Reversible MCTS in KV-Cache Latent Space** — MCTS that mutates KV-cache tensors in-place
   (`K_t ← K_t + α·d_K`) then reverses exactly via FP32 accumulators. Reduces memory from
   `O(b^d · M_KV)` to `O(M_KV + d·K_acc)`. Enables deep MCTS on a single H100 for 20B models.

3. **Orthogonal Escape Formalization** — Formal definition + experimental protocol for deceptive
   encodings redistributing into null spaces of linear monitors under corrective pressure.
   Key metrics: OEI (Orthogonal Escape Index) and TDS (Telemetry Divergence Score).

---

## Repo Structure

```
logomesh/                   — core research package
  oracle.py                 — BaseModelClient ABC + OpenAIModelClient
  local_model.py            — LocalLlamaOracle: HF transformers, hidden states, KV cache API
                              generate_one_step(), get_kv_cache(), set_kv_cache()
  hneuron_monitor.py        — H-Neuron stress σ_H; score() scalar + score_per_layer() → [L]
  whitebox.py               — RepE probes + PerLayerHonestyProjector ρ_R^(l) + steering vecs
  telemetry_matrix.py       — TelemetryMatrix T_t ∈ ℝ^{2×L}, DiagnosticState, compute_node_reward
  orthogonal_escape.py      — NullSpaceProjector, OEICalculator (Eq. 10), TDSCalculator
  kv_mcts.py                — ReversibleMCTS, FP32Accumulator (Theorem 1), KVCacheNode, MCTSConfig
  search_policy.py          — UCB1 bandit (node selection reuse)
  payload_library.py        — PayloadEntry + PayloadLibrary (Phase 4: extend to research dataset)
  croissant_export.py       — Croissant 1.1 + RAI export helpers + schema-shape validation
  evidence_store.py         — structured per-run logging
  graders.py                — PluginGrader, RuleBasedGrader, CompositeGrader
  ablation.py               — AblationConfig (experiment toggles)
  threat_model.py           — ThreatModel, GoalTaxonomy, AttackSurface

scripts/
  probe_kv_cache_mutability.py — Phase 2 gate: validates in-place mutation + reversibility
  run_kv_mcts.py               — Phase 2 runner: ReversibleMCTS with T_t, OEI, TDS JSON output
  export_kv_mcts_to_croissant.py — Runtime artifact -> Croissant package exporter
  measure_lipschitz_drift.py   — Theorem 1 validation: FP32 accumulator vs naive bf16 drift
  run_offline_mcts.py          — Phase A text-generation MCTS (baseline)
  train_lat_probes.py          — LAT probe training (Phase 3: retrain for paper semantics)

tests/
  test_sage.py                 — logomesh module unit tests (no LLM calls)
  test_whitebox.py             — RepE / WhiteBoxEvaluator tests
  test_local_model_interface.py — Phase 2 LocalLlamaOracle KV cache interface tests
  test_phase2_modules.py       — TelemetryMatrix, OEI, TDS, FP32Accumulator, MCTS smoke

docs/
  NeurIPS/                  — paper drafts (canonical: 04.18.2026-NeurIPS-Research-Proposal-2.tex)
  reviews/                  — gap analysis + transition audit
  logs/                     — session logs
  dataset/                  — Croissant schema stub (Phase 4)
```

---

## Phase Status

| Phase | Description | Status |
|---|---|---|
| 1 | Repo cleanup — `logomesh/` package, `BaseModelClient` interface, deleted competition code | ✅ Complete |
| A | Local 1B model offline MCTS foundations (H-Neurons, LAT probes, payload library) | ✅ Foundations built |
| 2 | Reversible KV-MCTS — `kv_mcts.py`, `telemetry_matrix.py`, `orthogonal_escape.py`, per-layer telemetry | ✅ Complete |
| 3 | Experiment infrastructure (5 experiment scripts, Procrustes, evaluation framework) | 🔲 Next |
| 4 | Research dataset (Croissant), paper writing | 🔲 Not started |

**Phase 3 gate: PASSED (2026-04-16) on `meta-llama/Llama-3.2-1B-Instruct`.**
All three Chunk 0 steps completed: KV-cache mutability probe ✓, 10-node MCTS smoke test ✓,
Lipschitz drift validation (200 cycles, FP32 accumulator exact at 0.00e+00) ✓.
Full record: `docs/logs/2026-04-16_session-log.md`.

---

## Hardware

| Environment | Hardware | Models |
|---|---|---|
| Phase 2 dev | RTX 3060 12GB | TinyLlama / Llama-3.2-1B-Instruct (~2GB VRAM) |
| Phase 3 experiments | Tianyu Shi's lab — 8× H100 80GB | gpt-oss-20b (MoE, ~16GB floor) |

---

## Key Models

| Model | Use | Access |
|---|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | Phase 2 KV-MCTS prototype | HuggingFace |
| `[7B TBD]` | Procrustes transfer target (Experiment 5) | TBD |
| `openai/gpt-oss-20b` | Phase 3 scaling (MoE, H100 only) | HuggingFace (Apache 2.0) — **TBC: not yet accessed** |

**gpt-oss-20b gotcha:** MoE (32 experts, 4 active/token). Use router logit entropy for
H-Neuron monitoring — do NOT use dense MLP neuron monitoring. See `hneuron_monitor.py`.
**Note:** Model availability and Apache 2.0 licence claim have not been verified against HuggingFace. Confirm before Phase 3 planning.

---

## Hard Rules

1. **No competition framing** in any code, comments, or docs. The paper is standalone.
2. **Run the full quality gate before every commit** (see Dev Commands below). All four checks
   must pass: ruff, mypy, bandit, pytest.
3. Phase 2 changes are **additive only** — never remove existing `logomesh/` module interfaces.
4. The paper's subject of monitoring is the **model under study itself** (detecting alignment
   faking in its own representations), not an external proxy model.
5. **Run the code review agent before completing any task** that modifies `logomesh/` or
   `scripts/` files. Do not present work as done until the review agent has passed. If it
   flags issues, fix them before committing. This is not optional.

---

## Paper-to-Code Verification (Mandatory)

Before modifying any `logomesh/` module:

1. **Read the paper equation** this code implements from the Key Paper Equations section below.
2. **Read the gap analysis doc** for the relevant contribution (see Gap Tracking below).
3. **After writing code**, show the mapping: paper symbol to code variable, paper operation
   to code operation. Example: "Eq. 3 `a_j^(l)` = `activation_row[i]`, `ReLU` = `max(0, ...)`".
4. **Flag any divergence** between paper and code explicitly. Do not silently approximate.

Reference files:
- `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` (canonical paper)
- `docs/contribution-1-telemetry-matrix.md` (C1 gaps)
- `docs/contribution-2-reversible-mcts.md` (C2 gaps)
- `docs/contribution-3-orthogonal-escape.md` (C3 gaps)

---

## Test Requirements

Every change to a `logomesh/` module must include or update a test that exercises the
**actual changed code path** with non-trivial inputs. Tests using `_Fake*` stubs only
verify downstream consumers — the changed module itself needs a test with realistic
mock data that flows through the real implementation.

For formula implementations (Eq. 3, 4, 5, 8, 10, etc.): include a test with known inputs
where the expected output is hand-calculated from the paper equation.

| Module | Test File | What Must Be Tested |
|---|---|---|
| `hneuron_monitor.py` | `test_phase2_modules.py` | `_calibrate_dense()` with per-layer tensor data, `score_per_layer()` with real tensors |
| `whitebox.py` | `test_whitebox.py` | `PerLayerHonestyProjector.project()` with tensor inputs |
| `telemetry_matrix.py` | `test_phase2_modules.py` | `classify()` threshold edge cases, `compute_node_reward()` against Eq. 8 |
| `orthogonal_escape.py` | `test_phase2_modules.py` | `OEICalculator.compute()` against Eq. 10 (delta formula) |
| `kv_mcts.py` | `test_phase2_modules.py` | `FP32Accumulator` apply/rollback with realistic shapes, `_read_telemetry()` integration |

---

## Gap Tracking

38 gaps between paper claims and code implementation are documented across three files.
These are the source of truth — do not duplicate gap data here.

Before modifying any `logomesh/` module, read the relevant gap analysis:
- Contribution 1 (T_t, sigma_H, rho_R): `docs/contribution-1-telemetry-matrix.md`
- Contribution 2 (MCTS, FP32, memory): `docs/contribution-2-reversible-mcts.md`
- Contribution 3 (OEI, TDS, NullSpace): `docs/contribution-3-orthogonal-escape.md`

When a gap is resolved, update the contribution doc to mark it done and note the gap ID
in the session log.

---

## Key Paper Equations (Quick Reference)

**Telemetry matrix** (Eq. 3):
`T_t = [[σ_H^(1)..σ_H^(L)], [ρ_R^(1)..ρ_R^(L)]]`

**MCTS reward** (Eq. 8):
`r(node) = λ₁·σ̄_H + λ₂·(1 - ρ̄_R) + λ₃·Divergence(σ_H, ρ_R)`

**OEI** (Eq. 10, revised April 21):
`OEI = ‖Proj_Null(P)(h_steered - h_base)‖₂ / ‖h_steered - h_base‖₂`
Range [0, 1]. OEI -> 1 = Orthogonal Escape. OEI -> 0 = normal response.

**Reversibility** (Eq. 6):
`A^(ℓ) ← A^(ℓ) + α·d^(ℓ)` (FP32), `K_t ← K_base + cast_bf16(A^(ℓ))`

---

## Dev Commands

```bash
# QUALITY GATE — run ALL FOUR before every commit (in this order):
ruff check logomesh/ scripts/                      # lint + style (must pass clean)
mypy logomesh/ --ignore-missing-imports            # type check (no new errors in changed files)
bandit -r logomesh/ -q                             # security scan (must pass clean)
uv run pytest tests/ -v                            # full test suite (must be 100% green)

# If any of the above fail on a file you changed, fix it before committing.

# Pre-commit hook (one-time setup — enforces the quality gate automatically):
pip install pre-commit && pre-commit install

# General
uv sync                                            # install/update deps

# Phase 2 runners — model must match paper (Llama-3.2-1B-Instruct, hidden dim 2048, 22 layers)
uv run python scripts/run_kv_mcts.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --nodes 50 --depth 5 --branches 3
uv run python scripts/measure_lipschitz_drift.py \
    --model meta-llama/Llama-3.2-1B-Instruct --n-cycles 200
uv run python scripts/probe_kv_cache_mutability.py --device auto

# Phase A model download (for run_offline_mcts.py and gate re-runs)
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir ./models/llama-3.2-1b
```
