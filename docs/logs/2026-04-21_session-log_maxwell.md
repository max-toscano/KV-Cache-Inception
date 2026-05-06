# Session Log — 2026-04-21 (Maxwell)

Session focus: Deep code-to-paper alignment analysis for all three contributions, gap identification, PR creation, and fix planning for Tier 1 gaps.

---

## 1. Code-to-Paper Mapping Documents Created

### What was done
Created three comprehensive markdown documents mapping every paper equation and claim to its code implementation, with deep analysis verifying alignment and identifying gaps.

### Files created

| File | Contribution | Gaps Found |
|---|---|---|
| `docs/contribution-1-telemetry-matrix.md` | C1: Multi-Dimensional Micro-Telemetry Matrix | 13 (1 HIGH, 5 MEDIUM, 7 LOW) |
| `docs/contribution-2-reversible-mcts.md` | C2: Reversible MCTS in KV-Cache Latent Space | 13 (0 HIGH, 5 MEDIUM, 8 LOW) |
| `docs/contribution-3-orthogonal-escape.md` | C3: Formalization of Orthogonal Escape | 12 (2 HIGH, 4 MEDIUM, 6 LOW) |

### Why
To verify the codebase faithfully implements what the paper claims before NeurIPS submission (May 6, 2026). Each document cross-references paper equations to exact file:line locations, identifies where code diverges from paper claims, rates severity, and proposes fixes.

### Total gaps identified: 38
- **5 HIGH** — blocking for paper claims or experiments
- **17 MEDIUM** — divergences that weaken the paper but don't block submission
- **16 LOW** — minor issues, config changes, or paper-only clarifications

---

## 2. PR Created

### Branch
`contributions-1.1-notes-for-code`

### Commit
`54015e8` — `docs: add code-to-paper mapping for all three contributions`

### What was pushed
- `docs/contribution-1-telemetry-matrix.md` (new)
- `docs/contribution-2-reversible-mcts.md` (new)
- `docs/contribution-3-orthogonal-escape.md` (new)

### PR link
Created via GitHub web UI at:
`https://github.com/max-toscano/KV-Cache-Inception/pull/new/contributions-1.1-notes-for-code`

### Why
To get team review on the gap analysis before starting fixes. Critical gaps (C1-04, C3-04, C3-05) affect experiment feasibility and need team alignment on priority.

---

## 3. Gap Fix Planning — Tier 1 (Telemetry Matrix)

Produced detailed implementation plans for the two Tier 1 (trivial) Contribution 1 gaps using planning agents. These are ready to implement.

### GAP-C1-11: Step index always zero in MCTS telemetry
- **Problem:** `kv_mcts.py:643` hardcodes `step=0` in `_read_telemetry()`.
- **Plan:** Add `step: int` parameter to `_read_telemetry()`, pass `n_expanded` from the MCTS loop. 3 lines changed.
- **Files:** `logomesh/kv_mcts.py` (lines 589, 626, 643)
- **Risk:** Zero — `step` is metadata only, no behavioral change.
- **Status:** Planned, not yet implemented.

### GAP-C1-02: Per-neuron ReLU not applied
- **Problem:** `hneuron_monitor.py:247-251` averages raw activations. Paper Eq. 3 requires `max(0, a_j - a_bar_j)` per neuron before averaging.
- **Plan:** Store per-neuron baselines during `_calibrate_dense()`, apply ReLU subtraction in `_raw_dense_score()`. ~10 lines changed.
- **Files:** `logomesh/hneuron_monitor.py` (lines 78, 233, 247-251)
- **Risk:** Low — calibration scores adapt automatically since both calibration and runtime use the same formula.
- **Status:** Planned, not yet implemented.

---

## 4. Code Changes This Session

**No production code was modified this session.** All work was analysis, documentation, and planning.

| Action | Files | Type |
|---|---|---|
| Created `docs/contribution-1-telemetry-matrix.md` | New file | Documentation |
| Created `docs/contribution-2-reversible-mcts.md` | New file | Documentation |
| Created `docs/contribution-3-orthogonal-escape.md` | New file | Documentation |
| Committed and pushed to `contributions-1.1-notes-for-code` | Git | Branch + commit |

---

## 5. Key Findings Summary

### Strongest alignment (code matches paper exactly)
- TelemetryMatrix T_t shape (2xL) — Eq. 5
- FP32 Accumulator apply/rollback — Eq. 6, Theorem 1
- MCTS node reward function — Eq. 8
- UCB1 selection — Eq. 9
- OEI formula — Eq. 10
- All four DiagnosticState classifications — Table 1

### Most critical gaps
- **C1-04 (HIGH):** MoE per-layer sigma_H returns all 0.5 — disables bottom-up channel for Phase B
- **C3-04 (HIGH):** Textual compliance rate not measured — can't test full Orthogonal Escape hypothesis
- **C3-05 (HIGH):** No Experiment 1 analysis script — data collected but hypothesis never tested
- **C2-03 (MEDIUM):** Full-shape accumulators — paper claims ~50MB overhead, actual is ~80GB for 20B model

### Estimated fix timeline
- Tier 1 (trivial): 1 session
- Tier 2 (small): 2-3 sessions
- Tier 3 (moderate): 4-5 sessions
- Tier 4 (significant): 7-10 sessions
- Total for submission-critical gaps: ~5-7 days

---

## Next Session Plan
1. Implement GAP-C1-11 (step counter) and GAP-C1-02 (per-neuron ReLU)
2. Run tests to verify no regressions
3. Move to Tier 2 gaps (C1-08, C1-13, C1-06, C1-03)
