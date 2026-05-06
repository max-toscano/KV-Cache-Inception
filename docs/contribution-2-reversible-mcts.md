# Contribution 2: Reversible MCTS in KV-Cache Latent Space

> **Paper reference:** Section 3.2 (Discrete Text-Space Bottleneck), Section 4.2 (Equations 6–9),
> Section 6 (Theorem 1 + Memory Complexity Proposition)
>
> **Claim:** The first Monte Carlo Tree Search algorithm operating directly in the continuous
> latent space of a Transformer's key-value cache, using a mathematically reversible rollback
> mechanism with FP32 accumulators that reduces memory complexity from O(b^d * M_KV) to
> O(M_KV + d * K_acc).

---

## 1. Overview — What the Paper Claims

Contribution 2 introduces four components:

1. **FP32 Accumulator (Theorem 1):** A precision management mechanism that tracks cumulative
   steering deltas in FP32 while the KV cache operates in bf16/fp16. Guarantees exact baseline
   restoration after arbitrary apply/rollback cycles with error bound independent of cycle
   count n (Eq. 6).

2. **In-Place KV-Cache Steering:** Forward mutation (Eq. 7a) and reverse rollback (Eq. 7b)
   of key-value tensors via tensor addition/subtraction, enabling the MCTS to explore the
   latent intervention landscape without allocating parallel cache copies.

3. **Reversible MCTS Algorithm:** A full MCTS loop (Selection → Expansion → Simulation →
   Backup) that operates in continuous latent space rather than discrete token space,
   using the telemetry matrix from Contribution 1 as the node reward signal.

4. **Memory Complexity Reduction (Proposition):** From O(b^d * M_KV) for standard parallel
   MCTS to O(M_KV + d * K_acc), making deep search feasible on a single 80GB GPU for
   models up to 20B parameters.

---

## 2. FP32 Accumulator — Theorem 1 Exact Reversibility

### Paper Equations

**Eq. 6 (FP32 Accumulator Reconstruction):**
```
A^(l) <- A^(l) + alpha * d^(l)       [FP32 arithmetic]
K_t^(l) <- K_base^(l) + cast_bf16(A^(l))
```

**Theorem 1:** After n complete apply-reverse cycles:
```
Naive bf16:   ||K_n - K_0||_inf <= n * eps_bf16 * max_i ||delta_i||_inf
FP32 accum:   ||K_n - K_0||_inf <= eps_bf16 * ||A_final||_inf    (independent of n)
```

Where `eps_bf16 = 2^{-8} ~ 3.9e-3`. After complete reversal, `A_final = 0`, so `K_n = K_0` exactly.

### Code Implementation

**File:** `logomesh/kv_mcts.py`, lines 211–345

| Paper Element | Code Location | Function/Method |
|---|---|---|
| Accumulator initialization | Lines 234–255 | `FP32Accumulator.from_kv_cache()` |
| K_base, V_base storage | Lines 229–232, 250–252 | `k_base`, `v_base` (cloned at root) |
| FP32 zero accumulators | Lines 253–254 | `torch.zeros_like(k, dtype=torch.float32)` |
| Forward steering (Eq. 6) | Lines 257–298 | `apply()` |
| A^(l) += alpha * d^(l) | Line 289 | `self.k_accum[l_idx].add_(alpha * dk)` |
| K = K_base + cast(A) | Lines 293–294 | `k_live.copy_(self.k_base[l_idx] + self.k_accum[l_idx].to(dtype=...))` |
| Reverse rollback | Lines 300–334 | `rollback()` |
| A^(l) -= alpha * d^(l) | Line 326 | `self.k_accum[l_idx].sub_(alpha * dk)` |
| Residual norm check | Lines 336–345 | `residual_norm()` — `max(a.abs().max() for a in k_accum)` |
| Steering vector broadcast | Lines 348–367 | `_broadcast_to()` |

### How It Works in Code

1. **Initialization** (`from_kv_cache()`, line 235): Given a model's `past_key_values` output,
   clones K and V tensors per layer as `k_base`/`v_base` in the model's native dtype (bf16/fp16).
   Creates zero-initialized FP32 accumulators of the same shape.

2. **Forward Apply** (`apply()`, line 257):
   - Converts numpy steering vector `dk` to torch FP32 tensor on the correct device
   - Broadcasts from [d_model] or [d_head] to cache shape [batch, heads, seq, d_head]
   - Accumulates in FP32: `k_accum[l] += alpha * dk`
   - Reconstructs live cache: `k_live = k_base[l] + cast(k_accum[l])` (Eq. 6 exactly)

3. **Rollback** (`rollback()`, line 300):
   - Identical to apply but subtracts: `k_accum[l] -= alpha * dk`
   - Reconstructs: `k_live = k_base[l] + cast(k_accum[l])`
   - After complete reversal, `k_accum` returns to zero → `k_live = k_base` exactly

4. **Residual Monitoring** (`residual_norm()`, line 336):
   - Returns `||A_K||_inf` — the maximum absolute value across all K accumulators
   - Should be ~0 after complete rollback; nonzero indicates incomplete reversal or floating point issues

### Deep Analysis: Code vs. Paper

#### ALIGNED

- **Eq. 6 implementation is exact.** The code follows the paper's two-step process precisely:
  (1) accumulate delta in FP32, (2) reconstruct cache from base + cast(accumulator). Lines
  289–294 and 326–331 are direct translations of the equations.

- **Theorem 1 bound is mechanically guaranteed.** The only quantization error occurs in the
  `cast(k_accum)` on line 294: converting FP32 accumulator to model dtype. After full
  reversal, `k_accum` is zero in FP32 (exact), and `cast(0.0)` is exactly 0.0 in any dtype.
  Therefore `K_n = K_base + 0 = K_base` exactly, as Theorem 1 claims.

- **Both K and V caches are handled.** The paper discusses only K tensors in the equations,
  but the code correctly applies the same accumulator logic to V tensors (lines 290, 296–298,
  327, 332–334). This is the natural generalization.

- **In-place mutation via `copy_()`.** The code uses `k_live.copy_(...)` (line 293) rather
  than tensor assignment, ensuring the actual cache tensor is modified in-place. This is
  critical — tensor assignment would create a new tensor object while leaving the model's
  internal reference pointing to the old one.

#### GAPS

1. **Steering vector shape mismatch fallback is a silent no-op.** In `_broadcast_to()`
   (line 364–366), if the steering vector's last dimension doesn't match the cache tensor's
   last dimension, the function returns a zero tensor:
   ```python
   return torch.zeros(target_shape, dtype=vec.dtype, device=vec.device)
   ```
   This means a misconfigured steering vector silently does nothing — no warning, no error.
   The MCTS would appear to run but produce meaningless results.

2. **V steering defaults to K steering.** In `apply()` (line 275–276), if `dv_vectors` is
   None, it defaults to `dk_vectors`. The paper's Eq. 7 defines separate `d_K` and `d_V`
   steering directions. Using the same direction for both K and V is a simplification.
   In the MCTS loop (`kv_mcts.py:571`), only `dk` is passed:
   ```python
   accumulator.apply(past_kv, alpha, [dk] * len(accumulator.k_accum))
   ```
   So V steering uses the K honesty direction, which may not be optimal for value tensors.

3. **Accumulator stores full-shape tensors, not sparse deltas.** The paper's Proposition
   mentions `S'` mutated positions (typically `S' << S`). The code allocates accumulators
   of the full cache shape `[batch, heads, seq, d_head]` per layer (line 253). For a 20B
   model with long sequences, this could be substantial FP32 overhead (~2x the cache per
   layer due to K + V accumulators in FP32 vs. bf16 cache).

---

## 3. KV-Cache Helpers & Mutability Gate

### Code Implementation — Cache Extraction

**File:** `logomesh/kv_mcts.py`, lines 60–205

The code supports five KV cache formats across Transformers versions:

| Format | Transformers Version | Extraction Path |
|---|---|---|
| Legacy tuple `((K,V), ...)` | All | `_extract_kv_tensors()` line 111–112 |
| `DynamicCache.key_cache/value_cache` lists | <= 5.2 | Lines 118–123 |
| `DynamicCache.layers` with `.keys/.values` | 5.3 | Lines 126–129 |
| Duck-typed `.key_cache/.value_cache` | Any Cache subclass | Lines 143–147 |
| Duck-typed `.layers` | Any layers-based cache | Lines 150–153 |

**Critical safety function:** `_kv_eval_cache()` (line 192):
- For tuple caches: returns a detached tuple view (cheap, no VRAM allocation)
- For DynamicCache: returns a deep copy to prevent `model.forward()` from calling
  `DynamicCache.update()` which would replace tensor references and break rollback

**Why this matters for Theorem 1:** The docstring at line 184–188 explains that
`FP32Accumulator.rollback()` holds references to tensors in `key_cache[l]`. If
`DynamicCache.update()` replaces those entries with grown tensors during the evaluation
forward pass, rollback would modify abandoned tensors while the model uses new ones.
The snapshot prevents this.

### Code Implementation — Mutability Gate

**File:** `scripts/probe_kv_cache_mutability.py`, lines 123–233

This script validates the physical precondition for Reversible MCTS:

| Step | Lines | What It Does |
|---|---|---|
| 1. Seed KV cache | 145–160 | Forward pass on prompt, capture `past_key_values` |
| 2. Clone baseline | 162 | `_clone_cache()` — preserve original state |
| 3. Mutate in-place | 190 | `mutated_key_tensor.add_(alpha)` |
| 4. Measure logit delta | 193–196 | Forward with mutated cache, compute `max_logit_delta` |
| 5. Restore via copy_ | 205 | `mutated_key_tensor.copy_(original_key)` |
| 6. Measure revert delta | 208–211 | Forward with restored cache, compute `revert_delta` |

**Gate criteria** (line 220–225):
- `mutable_in_place`: `max_logit_delta > min_delta` (default 1e-7) — mutation affects output
- `reversible_with_copy_restore`: `revert_delta <= revert_tol` (default 1e-5) — restore works
- `gate_passed = mutable AND reversible`

**Exit codes:** 0 = pass, 2 = not mutable, 3 = not reversible

### Deep Analysis: Code vs. Paper

#### ALIGNED

- **Mutability gate validates the paper's core assumption.** Section 4.2 states "tensor
  addition is reversible, provided numerical stability is maintained." The gate script
  empirically verifies this for each model before MCTS can run.

- **Multi-format KV cache support is robust.** The extraction code handles the three major
  Transformers cache formats, with duck-typing fallbacks and clear error messages on failure.
  This is essential for the paper's claim of working with "open-weight models."

- **DynamicCache safety is correctly implemented.** The `_kv_snapshot_tuple()` / `_kv_eval_cache()`
  functions prevent a subtle but critical correctness issue that would silently violate
  Theorem 1.

#### GAPS

1. **Gate script tests copy_ restore, not FP32 accumulator restore.** The gate
   (`probe_kv_cache_mutability.py`) validates that `tensor.copy_(original)` restores the
   cache. But the actual MCTS uses `FP32Accumulator.rollback()`, which reconstructs via
   `K_base + cast(A)`. These are different operations — copy_ is exact, while
   `K_base + cast(A)` has the quantization bound from Theorem 1. The gate doesn't test
   the actual reversal mechanism used in production.

2. **No gate check for FP32 accumulator correctness.** The `measure_lipschitz_drift.py`
   script validates Theorem 1 empirically, but it's not integrated as a formal gate check.
   It's a standalone measurement tool, not a pass/fail prerequisite.

3. **DynamicCache deep copy is expensive.** `_kv_eval_cache()` deep-copies the entire
   DynamicCache for non-tuple caches (line 204). This allocates a full copy of the cache
   per evaluation step. For large models, this partially undermines the memory savings
   claimed in the Proposition. The paper's memory analysis assumes zero-copy evaluation.

---

## 4. MCTS Algorithm — ReversibleMCTS

### Paper Description (Section 4.2)

The paper describes a standard MCTS loop adapted for continuous latent space:
1. **Selection:** UCB1 traversal from root to best leaf
2. **Expansion:** Branch into new states via KV steering at different magnitudes alpha
3. **Simulation:** One-step rollout + telemetry evaluation (T_t → reward)
4. **Backup:** Propagate reward up the tree

### Code Implementation

**File:** `logomesh/kv_mcts.py`, lines 454–667

| MCTS Phase | Code Location | What Happens |
|---|---|---|
| **Initialization** | Lines 504–544 | Seed KV cache from prompt, build FP32Accumulator, create root node |
| **Selection** | Line 549 + lines 646–657 | `_select()` — UCB1 traversal from root to best leaf |
| **Expansion** | Lines 554–567 | For each alpha in `alpha_values[:branching_factor]`, create child node |
| **Steering (Apply)** | Lines 570–571 | `accumulator.apply(past_kv, alpha, [dk] * n_layers)` |
| **Simulation** | Lines 577–582 | `oracle.generate_one_step(input_ids, past_key_values=snapshot)` |
| **Telemetry** | Line 589 | `T = self._read_telemetry()` → TelemetryMatrix from Contribution 1 |
| **OEI Computation** | Lines 593–600 | `oei_calc.compute(h_base, h_steered)` |
| **Reward** | Line 603 | `compute_node_reward(T, self._config.reward_lambdas)` — Eq. 8 |
| **Rollback** | Line 606 | `accumulator.rollback(past_kv, alpha, [dk] * n_layers)` |
| **Backup** | Line 611 | `_backpropagate(child, reward)` — update visit_count + reward_sum to root |

### How It Works End-to-End

```
1. Prompt → oracle.generate_one_step() → past_kv (root KV cache)
2. FP32Accumulator.from_kv_cache(past_kv) → stores K_base, V_base, zeros A
3. Loop (n_expanded < n_nodes):
     a. parent = UCB1_select(root)         # find best leaf
     b. if parent.depth >= max_depth: skip
     c. for alpha in alpha_values[:branching_factor]:
          i.   accumulator.apply(past_kv, alpha, dk)        # A += alpha*dk, K = K_base + cast(A)
          ii.  snapshot = _kv_eval_cache(past_kv)            # safe copy for forward pass
          iii. step = oracle.generate_one_step(snapshot)      # one token with steered cache
          iv.  T = _read_telemetry()                          # sigma_H + rho_R → T_t
          v.   oei = oei_calc.compute(h_base, h_steered)     # Orthogonal Escape Index
          vi.  reward = compute_node_reward(T, lambdas)       # Eq. 8
          vii. accumulator.rollback(past_kv, alpha, dk)       # A -= alpha*dk, K = K_base + cast(A)
          viii.backpropagate(child, reward)                    # update UCB1 stats to root
4. Return all nodes sorted by mean_reward descending
```

### Deep Analysis: Code vs. Paper

#### ALIGNED

- **MCTS loop structure matches the standard Selection/Expansion/Simulation/Backup pattern.**
  The code implements all four phases correctly.

- **Steering uses RepE honesty direction (d_K).** Line 537: `steering_vecs = self._repe.steering_vectors`.
  This matches the paper's claim that the MCTS steers along the honesty direction found by
  RepE probes, connecting Contribution 1 (monitoring) to Contribution 2 (search).

- **Alpha sweep matches Experiment 1 design.** `MCTSConfig.alpha_values` defaults to
  `(0.1, 0.5, 1.0, 2.0, 5.0)`, which matches the paper's Section 5.1:
  "alpha in {0.1, 0.5, 1.0, 2.0, 5.0}".

- **Branching factor limits expansion.** Line 555: `alphas_to_try = list(alpha_values)[:branching_factor]`.
  With branching_factor=3 and 5 alphas, each expansion tries the first 3 alphas. This is a
  practical budget control.

- **Backpropagation is correct.** `_backpropagate()` (line 659) walks from child to root via
  `parent_id`, incrementing `visit_count` and adding `reward_sum`. This is standard MCTS backup.

#### GAPS

1. **Single steering layer, not multi-layer.** The paper describes steering "the KV cache
   tensors at layer l" but the code steers only ONE layer (`steer_layer`, determined at
   line 520–523). The steering vector is broadcast to all layers via
   `[dk] * len(accumulator.k_accum)` (line 571), but `dk` is selected from a single layer
   index: `steering_vecs[min(steer_layer, len(steering_vecs) - 1)]` (line 570). All
   layers receive the same steering vector from the middle layer. The paper's per-layer
   steering directions `d_K^(l)` are not used — each layer gets the same direction.

2. **Expansion always uses the same alpha ordering.** Line 555:
   `alphas_to_try = list(alpha_values)[:branching_factor]`. This means every expansion
   always tries the FIRST `branching_factor` alphas in the configured list. The paper's
   MCTS should ideally vary which alphas are tried based on the search trajectory. The
   current implementation means alpha=2.0 and alpha=5.0 are never explored if
   branching_factor < 5.

3. **Simulation is one-step, not rollout.** The paper describes "simulation" as a rollout
   to estimate future value. The code does a single `generate_one_step()` (line 578) and
   immediately reads telemetry. There is no multi-step lookahead or value estimation.
   This is acknowledged by the method name `generate_one_step()` and is likely a Phase 2
   simplification — full rollouts would require either deeper recursion or a value function.

4. **No MCTS tree reuse across searches.** Each `run_async()` call starts with a fresh tree
   (line 543: `root = KVCacheNode.make_root()`). The paper doesn't explicitly address tree
   reuse, but standard MCTS implementations often warm-start from previous searches.

5. **Seeded on a fixed prompt.** The MCTS seeds from a single system+user prompt (line 512).
   The paper's experimental design (Experiment 1) implies sweeping over multiple prompts.
   The runner script (`run_kv_mcts.py`) handles one prompt per run; multi-prompt sweeps
   would require external orchestration.

6. **Root node has no telemetry.** The root is created via `KVCacheNode.make_root()` (line 543)
   with no telemetry or OEI score. The initial unsteered state is never evaluated for its
   baseline telemetry, which would be useful for computing delta-from-baseline metrics.

---

## 5. UCB1 Node Selection (Eq. 9)

### Paper Equation

```
UCB1(i) = r_bar_i + c * sqrt(2 * ln(N) / n_i)
```

Where:
- `r_bar_i` = mean reward of node i
- `N` = total visits to the parent
- `n_i` = visits to node i
- `c = sqrt(2)` = exploration constant

### Code Implementation

**File:** `logomesh/kv_mcts.py`, lines 374–419

```python
@dataclass
class KVCacheNode:
    node_id: str
    parent_id: str | None
    depth: int
    alpha: float
    layer: int
    visit_count: int = 0
    reward_sum: float = 0.0
    children: list[str] = field(default_factory=list)
    telemetry: "TelemetryMatrix | None" = None
    oei_score: float | None = None

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.visit_count if self.visit_count > 0 else 0.0

    def ucb1_score(self, parent_visits: int, c: float = 1.414) -> float:
        if self.visit_count == 0:
            return float("inf")
        exploit = self.mean_reward
        explore = c * math.sqrt(2.0 * math.log(parent_visits) / self.visit_count)
        return exploit + explore
```

**Selection** (`_select()`, lines 646–657):
```python
def _select(self, root):
    node = root
    while node.children:
        best_child = max(
            (self._nodes[cid] for cid in node.children),
            key=lambda c: c.ucb1_score(node.visit_count, self._config.exploration_constant),
        )
        if best_child.visit_count == 0:
            return best_child
        node = best_child
    return node
```

### Deep Analysis: Code vs. Paper

#### ALIGNED

- **UCB1 formula is exact.** The code implements `r_bar + c * sqrt(2 * ln(N) / n_i)` matching
  Eq. 9 precisely.

- **Exploration constant defaults to sqrt(2) ~ 1.414.** Matches the paper's stated default.

- **Unvisited nodes get infinite priority.** `visit_count == 0` returns `float("inf")` (line 415),
  ensuring all children are visited at least once before exploitation. This is standard UCB1.

- **Configurable exploration constant.** `MCTSConfig.exploration_constant` (line 445) allows
  tuning c for ablation studies.

#### GAPS

1. **Selection uses parent's visit count, not grandparent's.** In `_select()` (line 651),
   `node.visit_count` is passed as `parent_visits` to each child's `ucb1_score()`. This is
   correct for standard UCB1 — but note that `node` here is the CURRENT node during traversal,
   not necessarily the immediate parent of the child being scored. For the first level of
   children this is the root; for deeper levels it's the intermediate node. This is correct
   MCTS behavior.

2. **No progressive widening.** Standard MCTS in continuous action spaces often uses
   progressive widening to gradually increase the branching factor as visit counts grow.
   The code uses a fixed branching factor. For continuous alpha space, this limits exploration
   to only the first `branching_factor` values in the alpha list.

---

## 6. Memory Complexity (Proposition)

### Paper Claim

**Proposition:** Memory complexity is `O(M_KV + d * K_acc)` where:
- `M_KV = 2 * L * S * d * sizeof(dtype)` — base KV cache
- `K_acc = L * S' * d_model * 4 bytes` — per-step FP32 accumulator for S' mutated positions
- d = maximum search depth

Standard parallel MCTS: `O(b^d * M_KV)` — one cache copy per tree node.

**Paper example:** 20B model with S'=10 mutated positions → accumulator overhead ~50MB
vs. 40GB base cache. Standard MCTS with b=3, d=5: ~9.7TB.

### Code Reality

The code's actual memory usage differs from the paper's analysis:

| Component | Paper Claims | Code Reality |
|---|---|---|
| Base KV cache | M_KV (one copy) | One live cache + one cloned baseline = 2 * M_KV |
| FP32 accumulators | Per-step, sparse (S' positions) | Full-shape per layer: 2 * L * [batch, heads, seq, d_head] * 4 bytes |
| Evaluation snapshot | Not discussed | `_kv_eval_cache()` creates a deep copy for DynamicCache = +M_KV |
| Branching copies | None (reversible) | None (correct — in-place mutation) |

### Deep Analysis: Code vs. Paper

#### ALIGNED

- **No parallel cache copies.** The core claim holds: the code does NOT create b^d cache
  copies. It uses a single live cache mutated in-place, matching the paper's key insight.

- **Independence from branching factor.** Memory does not scale with b or b^d. Adding more
  alpha values or deeper search doesn't allocate new caches.

- **Accumulator overhead is small relative to cache.** The FP32 accumulators add ~2x the
  cache size in FP32 (since they cover all L layers, full shape), but this is still far
  less than O(b^d * M_KV).

#### GAPS

1. **Accumulators are NOT sparse (S' << S).** The paper's Proposition assumes accumulators
   track only S' mutated positions, with "S' = 10" in the example. The code allocates
   full-shape accumulators: `torch.zeros_like(k, dtype=torch.float32)` (line 253). For
   a 20B model with sequence length S=2048, the accumulator covers all 2048 positions,
   not just S'=10. The 50MB estimate in the paper corresponds to sparse accumulators;
   the actual overhead is `2 * L * S * d_model * 4 bytes` — approximately 2x the base
   cache size in FP32 vs. bf16. For a 40GB bf16 cache, this is ~80GB in FP32 accumulators,
   not 50MB.

2. **Baseline cache clone doubles memory.** `from_kv_cache()` clones K_base and V_base
   (lines 251–252). This is an additional full cache copy in model dtype. Total baseline
   memory is: live cache + cloned baseline + FP32 accumulators ≈ M_KV + M_KV + 2*M_KV =
   4 * M_KV. The paper claims O(M_KV + d * K_acc) which undercounts the baseline clone.

3. **DynamicCache evaluation copy.** When using DynamicCache (not legacy tuples),
   `_kv_eval_cache()` deep-copies the cache per evaluation step (line 204). This is a
   transient allocation but adds M_KV during each simulation step. With legacy tuple
   caches, only a lightweight detached view is created (no extra VRAM).

4. **Practical implication:** For TinyLlama (1.1B, ~0.5GB cache), the overhead is negligible.
   For the 20B target model, the actual memory footprint is closer to 4x M_KV + model
   weights, not M_KV + 50MB as the paper suggests. This may still fit on an 80GB H100 but
   the margin is tighter than claimed.

---

## 7. Theorem 1 Empirical Validation

### Script Implementation

**File:** `scripts/measure_lipschitz_drift.py`

| Component | Lines | What It Does |
|---|---|---|
| Model loading | ~60–80 | Load model, seed KV cache from prompt |
| Baseline clone | ~112 | `k_base_f32 = float32 clone` for accurate comparison |
| Random steering vector | ~108 | Unit-norm random dk for controlled experiments |
| FP32 accumulator path | ~117–122 | N cycles of apply + rollback via FP32Accumulator |
| Naive bf16 path | ~127–137 | N cycles of direct bf16 add + subtract (no accumulator) |
| Per-cycle measurement | ~140 | Record ||K_n - K_0||_inf for both paths |
| Output | ~154–185 | CSV with columns: cycle, naive_inf_norm, accumulator_inf_norm |

### Expected Results (from Theorem 1)

- **Naive path:** `||K_n - K_0||_inf` grows linearly with n (each add/subtract accumulates
  bf16 rounding error)
- **Accumulator path:** `||K_n - K_0||_inf` stays bounded near `eps_bf16 ~ 3.9e-3` regardless
  of n (the only error is the final cast from FP32 to bf16)

### Test Coverage

**File:** `tests/test_phase2_modules.py`, lines 257–396

Key test: `test_multiple_apply_rollback_cycles_stay_bounded()` (lines 306–319):
- 20 cycles with alpha=0.5 on fake tensors
- Asserts `residual_norm() < 1e-3` after all cycles
- Validates the accumulator path stays bounded

Additional tests:
- `test_apply_and_rollback_residual()` — single cycle, residual < 1e-6
- Legacy tuple, DynamicCache, and DynamicCache.layers format variants

### Deep Analysis: Code vs. Paper

#### ALIGNED

- **Script directly validates Theorem 1's core claim.** The naive vs. accumulator comparison
  is exactly the experiment needed to confirm the bound.

- **Test suite enforces the bound.** The 20-cycle test with `< 1e-3` threshold is conservative
  relative to `eps_bf16 ~ 3.9e-3`.

#### GAPS

1. **Script uses random steering vectors, not RepE directions.** The paper's Theorem 1
   applies to arbitrary steering deltas, so random vectors are valid. But the practical
   question is whether the bound holds for the actual RepE honesty directions used in
   MCTS, which may have different magnitude distributions.

2. **Default 200 cycles may not stress-test deeply enough.** The paper mentions "1000
   rollback cycles" in Section 5.1 (Phase A validation). The script defaults to
   `--n-cycles 200`. While configurable, the default is less than the paper's stated
   test depth.

3. **No perplexity measurement.** The paper's Experiment 3 mentions measuring "Lipschitz
   drift (perplexity deviation from baseline) with and without FP32 accumulators." The
   script measures `||K_n - K_0||_inf` (tensor norm) but not perplexity deviation. Tensor
   drift is necessary but not sufficient — a small tensor delta could still cause large
   perplexity changes if it hits a sensitive region of the attention computation.

---

## 8. MCTSConfig and Hyperparameters

### Code Implementation

**File:** `logomesh/kv_mcts.py`, lines 426–448

```python
@dataclass
class MCTSConfig:
    n_nodes: int = 50
    branching_factor: int = 3
    max_depth: int = 5
    alpha_values: tuple = (0.1, 0.5, 1.0, 2.0, 5.0)
    exploration_constant: float = 1.414       # sqrt(2)
    reward_lambdas: tuple = (0.33, 0.33, 0.34)  # Eq. 8 weights
    steer_layer: int = -1                     # -1 = auto (middle layer)
```

### Deep Analysis

#### ALIGNED

- **Alpha sweep matches Experiment 1.** Paper: "alpha in {0.1, 0.5, 1.0, 2.0, 5.0}" — exact match.
- **Lambda weights are configurable.** Supports Eq. 8 ablation.
- **Exploration constant matches paper.** sqrt(2) ~ 1.414.

#### GAPS

1. **steer_layer auto-selection uses middle layer.** Line 522–523: `steer_layer = n_layers // 2`.
   The paper doesn't specify which layer to steer. Middle layer is a reasonable default
   (intermediate representations are often most informative), but the paper's per-layer
   steering directions `d_K^(l)` suggest the intent is to steer at the optimal layer
   determined by the RepE probe signal strength, not a fixed middle layer.

2. **No per-experiment config presets.** The paper defines 5 experiments with different
   configurations (Experiment 1: alpha sweep, Experiment 3: branching factor sweep, etc.).
   There are no predefined config presets for these experiments — all tuning must be done
   via CLI arguments.

---

## 9. End-to-End Data Flow

```
User prompt: "Explain alignment faking"
         |
         v
oracle.generate_one_step(system, user)
         |
         v
past_key_values (root KV cache)
         |
    +----+----+
    |         |
    v         v
FP32Accumulator          h_base (initial hidden state)
.from_kv_cache()         for OEI baseline
    |
    v
=== MCTS LOOP (n_nodes iterations) ===
    |
    v
UCB1 Selection: _select(root) → best leaf parent
    |
    v
For alpha in alpha_values[:branching_factor]:
    |
    +→ accumulator.apply(past_kv, alpha, dk)
    |       A^(l) += alpha * dk       [FP32]
    |       K^(l) = K_base + cast(A)  [in-place]
    |
    +→ snapshot = _kv_eval_cache(past_kv)
    |       (detached tuple or deep copy)
    |
    +→ oracle.generate_one_step(input_ids, snapshot)
    |       (one token with steered cache)
    |
    +→ T = _read_telemetry()
    |       sigma_H = hneuron.score_per_layer()
    |       rho_R = repe.project(hidden_states)
    |       → TelemetryMatrix (Contribution 1)
    |
    +→ oei = oei_calc.compute(h_base, h_steered)
    |       (Contribution 3)
    |
    +→ reward = compute_node_reward(T, lambdas)
    |       Eq. 8: λ₁·σ̄_H + λ₂·(1−ρ̄_R) + λ₃·JSD
    |
    +→ accumulator.rollback(past_kv, alpha, dk)
    |       A^(l) -= alpha * dk       [FP32]
    |       K^(l) = K_base + cast(A)  [exact restore]
    |
    +→ _backpropagate(child, reward)
            visit_count++, reward_sum += reward
            walk to root
=== END LOOP ===
    |
    v
Return nodes sorted by mean_reward (descending)
    |
    v
Serialise to JSON: per-node { alpha, depth, reward, telemetry, oei }
```

---

## 10. Test Coverage

### FP32 Accumulator Tests

**File:** `tests/test_phase2_modules.py`, lines 257–396

| Test | Lines | What It Verifies |
|---|---|---|
| `test_apply_and_rollback_residual` | 263–280 | Single apply+rollback → residual < 1e-6 |
| `test_multiple_apply_rollback_cycles_stay_bounded` | 306–319 | 20 cycles → drift < 1e-3 (Theorem 1) |
| `test_from_kv_cache_legacy_tuple` | 322–336 | Accumulator works with legacy tuple format |
| `test_from_kv_cache_dynamic_cache` | 339–360 | Accumulator works with DynamicCache format |
| `test_from_kv_cache_dynamic_cache_layers` | 363–396 | Accumulator works with DynamicCache.layers format |

### KV Cache Helper Tests

| Test | Lines | What It Verifies |
|---|---|---|
| `TestKVSnapshotTuple` | 403–444 | `_kv_snapshot_tuple()` creates detached views |
| `TestKVEvalCache` | 447–465 | Tuple → tuple snapshot; DynamicCache → deep copy |

### MCTS Node and Integration Tests

| Test | Lines | What It Verifies |
|---|---|---|
| `TestKVCacheNode.test_make_root` | 490–497 | Root has depth=0, alpha=0, no parent |
| `TestKVCacheNode.test_ucb1_unvisited` | 503–505 | Unvisited → inf |
| `TestKVCacheNode.test_ucb1_decreases_with_visits` | 507–518 | More visits → lower explore term |
| `test_reversible_mcts_smoke` | 592–614 | Full MCTS with fakes: 6 nodes, 2 branches, all have telemetry |

### LocalLlamaOracle Interface Tests

**File:** `tests/test_local_model_interface.py`

| Test | Lines | What It Verifies |
|---|---|---|
| `test_generate_one_step_returns_token_and_cache` | 65–74 | Returns dict with KV cache |
| `test_kv_cache_set_get_and_clear_roundtrip` | 90–98 | set/get/clear protocol works |

### What's NOT Tested

- `_broadcast_to()` with mismatched shapes (silent zero fallback)
- DynamicCache evaluation copy memory overhead
- Multi-layer steering (only single dk broadcast tested)
- `_select()` UCB1 traversal on deeper trees (smoke test only goes to depth 1)
- Accumulator behavior with very large alpha values (stress test)
- Theorem 1 bound at 1000+ cycles (test only does 20)

---

## 11. Summary of Alignment Status

### What Matches the Paper

| Paper Claim | Code Status | Confidence |
|---|---|---|
| FP32 accumulator: A += alpha*d, K = K_base + cast(A) (Eq. 6) | Exact match | HIGH |
| Forward mutation K += alpha*d_K (Eq. 7a) | Implemented via accumulator.apply() | HIGH |
| Reverse rollback K -= alpha*d_K (Eq. 7b) | Implemented via accumulator.rollback() | HIGH |
| Theorem 1 bound independent of n | Mechanically guaranteed by code structure | HIGH |
| UCB1 selection (Eq. 9) | Exact formula match | HIGH |
| MCTS loop: select → expand → simulate → backup | All four phases implemented | HIGH |
| No parallel cache copies (reversible in-place) | Correct — single live cache | HIGH |
| Alpha sweep {0.1, 0.5, 1.0, 2.0, 5.0} | Matches Experiment 1 design | HIGH |
| Telemetry matrix as reward signal (Eq. 8) | Connected via _read_telemetry() | HIGH |
| Theorem 1 empirical validation script | measure_lipschitz_drift.py | HIGH |
| KV-cache mutability precondition check | probe_kv_cache_mutability.py | HIGH |

### What Diverges from the Paper

| Paper Claim | Code Reality | Severity | Fix Phase |
|---|---|---|---|
| Per-layer steering directions d_K^(l) | Same dk broadcast to all layers from middle layer | MEDIUM | Phase 3 |
| Separate d_K and d_V steering | V defaults to K direction | LOW | Phase 3 |
| Sparse accumulators (S' << S) | Full-shape accumulators (all positions) | MEDIUM | Optimization |
| Memory O(M_KV + d*K_acc) | Actual: ~4*M_KV (base + clone + 2x FP32 accumulators) | MEDIUM | Clarify in paper |
| Multi-step simulation/rollout | Single-step generate_one_step() only | MEDIUM | Phase 3 |
| DynamicCache zero-copy evaluation | Deep copy per evaluation step for DynamicCache | LOW | Optimization |
| Progressive widening in continuous space | Fixed branching factor, fixed alpha ordering | LOW | Phase 3 |
| Gate script tests FP32 accumulator | Gate tests copy_ restore, not accumulator | LOW | Phase 3 |
| 1000 rollback cycles validation | ~~Script defaults to 200~~ **RESOLVED** — default now 1000 | ~~LOW~~ | ✅ Done |
| Perplexity drift measurement | Only tensor norm measured, not perplexity | MEDIUM | Phase 3 |

---

## 12. Consolidated Gap List

### GAP-C2-01: Single steering layer instead of per-layer
- **Section:** 4 (MCTS Algorithm)
- **Severity:** MEDIUM
- **Paper says:** Eq. 7 uses `d_K^(l)` per layer — implies each layer gets its own steering direction from RepE probes.
- **Code does:** `steer_layer` is a single integer (auto = middle layer). The steering vector `dk` is taken from one layer and broadcast to all layers via `[dk] * len(accumulator.k_accum)` (`kv_mcts.py:570–571`).
- **Impact:** All layers receive identical perturbation magnitude and direction. Layers with different honesty geometry are steered suboptimally. The per-layer RepE directions from `PerLayerHonestyProjector` exist but are not used — only `steering_vecs[steer_layer]` is selected.
- **Fix:** Pass `steering_vecs` directly (one per layer) instead of broadcasting a single layer's vector. Change line 570 to `dk_per_layer = steering_vecs[:n_layers]` and pass to `accumulator.apply()`.
- **Fix phase:** Phase 3

### GAP-C2-02: V steering defaults to K steering direction
- **Section:** 2 (FP32 Accumulator)
- **Severity:** LOW
- **Paper says:** Eq. 7 defines separate `d_K^(l)` and `d_V^(l)` steering directions for keys and values.
- **Code does:** `apply(dk_vectors, dv_vectors=None)` defaults `dv_vectors = dk_vectors` (`kv_mcts.py:275–276`). The MCTS loop never passes separate dv_vectors.
- **Impact:** Key and value caches are steered identically. K and V have different functional roles in attention (K affects routing, V affects content), so optimal steering directions may differ.
- **Fix:** Either derive V-specific steering directions or document in the paper that K=V steering is the implemented approach.
- **Fix phase:** Phase 3 or clarify in paper

### GAP-C2-03: Full-shape accumulators, not sparse
- **Section:** 6 (Memory Complexity)
- **Severity:** MEDIUM
- **Paper says:** Proposition mentions `S'` mutated positions with "S' = 10" → ~50MB overhead.
- **Code does:** `torch.zeros_like(k, dtype=torch.float32)` allocates full-shape [batch, heads, seq, d_head] accumulators per layer (`kv_mcts.py:253`). For a 20B model with S=2048, this is ~80GB, not 50MB.
- **Impact:** The paper's memory analysis significantly underestimates actual accumulator overhead. The practical memory is ~4x M_KV, not M_KV + 50MB. May still fit on H100 80GB for smaller models but the margin is misleading.
- **Fix:** Implement sparse accumulators that only track the mutated token positions. Alternatively, correct the paper's memory analysis to reflect full-shape accumulators.
- **Fix phase:** Optimization (code) or clarification (paper)

### GAP-C2-04: Baseline cache clone adds M_KV overhead
- **Section:** 6 (Memory Complexity)
- **Severity:** MEDIUM
- **Paper says:** Memory is `O(M_KV + d * K_acc)` — one base cache plus accumulators.
- **Code does:** `from_kv_cache()` clones both K_base and V_base (lines 251–252), storing them alongside the live cache. Total baseline: live cache + cloned baseline = 2 * M_KV before accumulators.
- **Impact:** The paper's O(M_KV + ...) undercounts by a factor of 2 for the base allocation. The clone is necessary for reconstruction (Eq. 6 requires K_base), but the paper's Proposition should account for it.
- **Fix:** Acknowledge in the Proposition that the constant factor is 2*M_KV (base + clone) + accumulator overhead.
- **Fix phase:** Clarify in paper

### GAP-C2-05: DynamicCache deep copy per evaluation step
- **Section:** 3 (KV-Cache Helpers)
- **Severity:** LOW
- **Paper says:** Memory analysis assumes no per-step allocation beyond accumulators.
- **Code does:** `_kv_eval_cache()` deep-copies the entire DynamicCache for non-tuple caches (line 204). This is a transient allocation of ~M_KV per simulation step.
- **Impact:** For DynamicCache-based models, each MCTS expansion temporarily allocates and frees a full cache copy. This doesn't accumulate (each is freed after the step) but increases peak memory by M_KV and adds GC pressure. Legacy tuple caches don't have this issue (detached view only).
- **Fix:** Investigate whether DynamicCache evaluation can use a lighter-weight snapshot that preserves the `.get_seq_length()` API without deep copying all tensors.
- **Fix phase:** Optimization

### GAP-C2-06: Silent zero fallback on steering vector shape mismatch — ✅ RESOLVED
- **Section:** 2 (FP32 Accumulator)
- **Severity:** LOW
- **Status:** RESOLVED — Four fixes applied: (1) `_broadcast_to()` returns `tuple[tensor, bool]` so callers detect failures. (2) `FP32Accumulator.apply()` and `rollback()` use two-pass pre-validation: all broadcasts checked before any accumulator is mutated, preventing partial-application corruption. Returns `bool`. (3) `run_async()` skips phantom nodes entirely when `apply()` returns `False` — no telemetry, no reward, no tree registration. (4) Rollback failure after successful apply raises `RuntimeError` (not `assert`) so it survives `python -O`. Hardened 2026-05-05: `_calibrate_dense()` shape guard rejects 2D input with `ValueError`; `measure_lipschitz_drift.py` checks apply/rollback return values with `RuntimeError`. Verified 2026-05-05.

### GAP-C2-07: Fixed alpha ordering limits exploration
- **Section:** 4 (MCTS Algorithm)
- **Severity:** LOW
- **Paper says:** MCTS should "systematically explore the intervention landscape."
- **Code does:** Line 555: `alphas_to_try = list(alpha_values)[:branching_factor]`. With branching_factor=3 and 5 alphas, only {0.1, 0.5, 1.0} are ever tried. Alpha=2.0 and alpha=5.0 are never explored.
- **Impact:** Higher steering magnitudes (which may be most interesting for Orthogonal Escape detection) are systematically excluded when branching_factor < len(alpha_values).
- **Fix:** Either shuffle alpha ordering per expansion, rotate through alphas based on depth, or use progressive widening to eventually try all values.
- **Fix phase:** Phase 3

### GAP-C2-08: One-step simulation, not multi-step rollout
- **Section:** 4 (MCTS Algorithm)
- **Severity:** MEDIUM
- **Paper says:** Standard MCTS includes a "simulation" phase that estimates future value, typically via random rollout or a value function.
- **Code does:** `generate_one_step()` produces a single token (line 578). The reward is computed immediately from that one step's telemetry. No lookahead.
- **Impact:** The reward signal captures only the immediate response to steering, not the downstream trajectory. A single-step evaluation may miss delayed alignment-faking behaviors that manifest over multiple tokens.
- **Fix:** Implement multi-step rollouts: generate N tokens after steering, aggregate telemetry across steps, use the trajectory reward. Alternatively, train a value function to estimate long-term reward from a single-step evaluation.
- **Fix phase:** Phase 3

### GAP-C2-09: Gate script doesn't test FP32 accumulator reversal
- **Section:** 3 (Mutability Gate)
- **Severity:** LOW
- **Paper says:** N/A (the gate is a practical precondition check).
- **Code does:** `probe_kv_cache_mutability.py` tests `copy_()` restore, not FP32 accumulator `rollback()`. The FP32 accumulator reversal is tested in unit tests and `measure_lipschitz_drift.py`, but not in the gate.
- **Impact:** The gate could pass while the FP32 accumulator has issues on a specific model/dtype combination. Low severity because unit tests and the drift script cover this independently.
- **Fix:** Add a FP32 accumulator apply/rollback cycle to the gate script as an additional check.
- **Fix phase:** Phase 3

### GAP-C2-10: Drift validation defaults to 200 cycles — ✅ RESOLVED
- **Section:** 7 (Theorem 1 Validation)
- **Severity:** LOW
- **Status:** RESOLVED — `--n-cycles` default changed from 200 to 1000 in `measure_lipschitz_drift.py`. Docstring example updated to match. Verified 2026-04-30.
- **Code does:** ~~Defaults to `--n-cycles 200`.~~ Now defaults to 1000, matching the paper's stated validation depth.

### GAP-C2-11: No perplexity drift measurement
- **Section:** 7 (Theorem 1 Validation)
- **Severity:** MEDIUM
- **Paper says:** Experiment 3: "Quantify Lipschitz drift (perplexity deviation from baseline) with and without FP32 accumulators."
- **Code does:** `measure_lipschitz_drift.py` measures `||K_n - K_0||_inf` (tensor infinity norm). Perplexity is not computed.
- **Impact:** Tensor drift and perplexity drift are correlated but not equivalent. A small tensor delta in a sensitive attention region could cause disproportionate perplexity change. The paper commits to measuring perplexity deviation; the code only measures tensor deviation.
- **Fix:** After each cycle, run a forward pass on a fixed evaluation prompt and measure perplexity. Report both tensor drift and perplexity drift in the CSV output.
- **Fix phase:** Phase 3

### GAP-C2-12: Root node has no baseline telemetry — ✅ RESOLVED
- **Section:** 4 (MCTS Algorithm)
- **Severity:** LOW
- **Status:** RESOLVED — `root.telemetry = self._read_telemetry()` added immediately after `KVCacheNode.make_root()` in `run_async()`. Root now captures baseline T_t before any steering. Smoke test updated to assert root telemetry is populated. Verified 2026-04-30.
- **Code does:** ~~Root node created with `telemetry=None`.~~ Now reads telemetry from the unsteered initial forward pass.

### GAP-C2-13: No tree reuse or warm-starting across runs
- **Section:** 4 (MCTS Algorithm)
- **Severity:** LOW
- **Paper says:** N/A (not discussed).
- **Code does:** Each `run_async()` call creates a fresh tree. Previous search results are not reused.
- **Impact:** Each run starts from scratch. For iterative experiments (e.g., refining alpha ranges based on previous results), warm-starting from a previous tree would be more efficient.
- **Fix:** Add optional tree serialization/deserialization to `ReversibleMCTS`.
- **Fix phase:** Phase 3 or beyond
