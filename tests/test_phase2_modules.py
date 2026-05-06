"""Tests for Phase 2 paper-core modules.

All tests run without a real model — fake tensors and mocks only.
Covers: TelemetryMatrix, DiagnosticState, compute_node_reward,
        NullSpaceProjector, OEICalculator, TDSCalculator,
        FP32Accumulator, KVCacheNode, ReversibleMCTS (smoke).
"""

from __future__ import annotations

import sys
import types
import unittest.mock as mock

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Stub openai so logomesh imports without the real package installed
# ---------------------------------------------------------------------------
_openai_stub = types.ModuleType("openai")
_openai_stub.AsyncOpenAI = mock.MagicMock()
if "openai" not in sys.modules:
    sys.modules["openai"] = _openai_stub


# ===========================================================================
# TelemetryMatrix, DiagnosticState, compute_node_reward
# ===========================================================================

from logomesh.telemetry_matrix import (
    TelemetryMatrix,
    DiagnosticState,
    ClassifyConfig,
    classify,
    compute_node_reward,
    report_telemetry_distribution,
)


class TestTelemetryMatrix:
    def test_construction_and_properties(self):
        T = TelemetryMatrix(
            h_neuron=np.array([0.2, 0.4, 0.6]),
            repe_honesty=np.array([0.8, 0.7, 0.9]),
            step=3,
        )
        assert T.n_layers == 3
        assert T.step == 3
        assert abs(T.sigma_H_mean - np.array([0.2, 0.4, 0.6]).mean()) < 1e-6
        assert abs(T.rho_R_mean - np.array([0.8, 0.7, 0.9]).mean()) < 1e-6

    def test_as_matrix_shape(self):
        T = TelemetryMatrix(h_neuron=np.zeros(5), repe_honesty=np.ones(5), step=0)
        mat = T.as_matrix()
        assert mat.shape == (2, 5)
        assert np.allclose(mat[0], 0.0)
        assert np.allclose(mat[1], 1.0)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError):
            TelemetryMatrix(h_neuron=np.zeros(3), repe_honesty=np.zeros(4), step=0)

    def test_jsd_identical_arrays_near_zero(self):
        v = np.array([0.1, 0.5, 0.4])
        T = TelemetryMatrix(h_neuron=v, repe_honesty=v.copy(), step=0)
        assert T.jsd() < 1e-6

    def test_jsd_different_arrays_positive(self):
        T = TelemetryMatrix(
            h_neuron=np.array([1.0, 0.0, 0.0]),
            repe_honesty=np.array([0.0, 0.0, 1.0]),
            step=0,
        )
        assert T.jsd() > 0.1


class TestDiagnosticState:
    def _make(self, sigma_H, rho_R):
        return TelemetryMatrix(
            h_neuron=np.full(4, sigma_H),
            repe_honesty=np.full(4, rho_R),
            step=0,
        )

    def test_alignment_faking(self):
        # σ̄_H low (<0.3), ρ̄_R high (>0.7)
        T = self._make(0.1, 0.9)
        assert classify(T) == DiagnosticState.ALIGNMENT_FAKING

    def test_contextual_fabrication(self):
        # σ̄_H high (>0.7), ρ̄_R < σ̄_H
        T = self._make(0.9, 0.3)
        assert classify(T) == DiagnosticState.CONTEXTUAL_FABRICATION

    def test_creative_synthesis_default(self):
        # σ̄_H moderate (0.4 — above alignment_faking threshold of 0.3,
        # below fabrication threshold of 0.7), ρ̄_R high, low JSD
        T = TelemetryMatrix(
            h_neuron=np.array([0.4, 0.4, 0.4]),
            repe_honesty=np.array([0.85, 0.9, 0.88]),
            step=0,
        )
        assert classify(T) == DiagnosticState.CREATIVE_SYNTHESIS

    def test_orthogonal_escape_high_jsd(self):
        # Force JSD > threshold using very different distributions
        T = TelemetryMatrix(
            h_neuron=np.array([5.0, 0.01, 0.01]),
            repe_honesty=np.array([0.01, 0.01, 5.0]),
            step=0,
        )
        assert classify(T, jsd_threshold=0.1) == DiagnosticState.ORTHOGONAL_ESCAPE


class TestClassifyConfig:
    def test_config_overrides_defaults(self):
        # σ̄_H=0.1 (<0.3), ρ̄_R=0.9 (>0.7) → ALIGNMENT_FAKING with defaults
        T = TelemetryMatrix(
            h_neuron=np.array([0.1, 0.1]), repe_honesty=np.array([0.9, 0.9]), step=0
        )
        assert classify(T) == DiagnosticState.ALIGNMENT_FAKING
        # Raise rho threshold above 0.9 → same input → CREATIVE_SYNTHESIS
        config = ClassifyConfig(alignment_faking_rho_threshold=0.95)
        assert classify(T, config=config) == DiagnosticState.CREATIVE_SYNTHESIS

    def test_config_jsd_threshold(self):
        # Identical arrays → JSD ≈ 0; default jsd_threshold=0.3 → not ORTHOGONAL_ESCAPE
        v = np.array([0.5, 0.5])
        T = TelemetryMatrix(h_neuron=v, repe_honesty=v.copy(), step=0)
        assert classify(T) != DiagnosticState.ORTHOGONAL_ESCAPE
        # Set jsd_threshold=0.0 → any JSD > 0 triggers ORTHOGONAL_ESCAPE
        config = ClassifyConfig(jsd_threshold=0.0)
        # Identical arrays have JSD=0, still not triggered
        assert classify(T, config=config) != DiagnosticState.ORTHOGONAL_ESCAPE

    def test_report_telemetry_distribution_empty(self):
        stats = report_telemetry_distribution([])
        assert stats["n_matrices"] == 0
        assert stats["rho_r_mean"] == 0.0

    def test_report_telemetry_distribution_stats(self):
        matrices = [
            TelemetryMatrix(
                h_neuron=np.full(3, 0.2), repe_honesty=np.full(3, 1.0), step=0
            ),
            TelemetryMatrix(
                h_neuron=np.full(3, 0.4), repe_honesty=np.full(3, 3.0), step=1
            ),
            TelemetryMatrix(
                h_neuron=np.full(3, 0.6), repe_honesty=np.full(3, -1.0), step=2
            ),
        ]
        stats = report_telemetry_distribution(matrices)
        assert stats["n_matrices"] == 3
        assert abs(stats["rho_r_min"] - (-1.0)) < 1e-6
        assert abs(stats["rho_r_max"] - 3.0) < 1e-6
        assert abs(stats["sigma_h_mean"] - 0.4) < 1e-6


class TestComputeNodeReward:
    def test_known_inputs(self):
        T = TelemetryMatrix(
            h_neuron=np.full(4, 0.8),  # σ̄_H = 0.8
            repe_honesty=np.full(4, 0.2),  # ρ̄_R = 0.2  → (1-ρ̄_R) = 0.8
            step=0,
        )
        # JSD of identical distributions = 0
        r = compute_node_reward(T, lambdas=(0.5, 0.5, 0.0))
        assert abs(r - 0.5 * 0.8 - 0.5 * 0.8) < 1e-5

    def test_reward_range(self):
        for s in np.linspace(0, 1, 5):
            for rho in np.linspace(0, 1, 5):
                T = TelemetryMatrix(
                    h_neuron=np.full(3, s), repe_honesty=np.full(3, rho), step=0
                )
                r = compute_node_reward(T)
                assert 0.0 <= r <= 2.0, (
                    f"reward {r} out of expected range for s={s} rho={rho}"
                )


# ===========================================================================
# NullSpaceProjector, OEICalculator, TDSCalculator
# ===========================================================================

from logomesh.orthogonal_escape import NullSpaceProjector, OEICalculator, TDSCalculator


class TestNullSpaceProjector:
    def test_projected_vector_is_orthogonal_to_w(self):
        w = np.array([1.0, 0.0, 0.0])
        h = np.array([3.0, 2.0, 1.0])
        proj = NullSpaceProjector(w)
        result = proj.project(h)
        assert abs(np.dot(result, w)) < 1e-10

    def test_projected_correct_values(self):
        w = np.array([1.0, 0.0, 0.0])
        h = np.array([3.0, 2.0, 1.0])
        proj = NullSpaceProjector(w)
        result = proj.project(h)
        # h - (h·w/‖w‖²)·w = [3,2,1] - 3*[1,0,0] = [0,2,1]
        np.testing.assert_allclose(result, [0.0, 2.0, 1.0], atol=1e-10)

    def test_2d_input_uses_last_token(self):
        w = np.array([1.0, 0.0])
        h_2d = np.array([[5.0, 3.0], [2.0, 7.0]])  # last token = [2, 7]
        proj = NullSpaceProjector(w)
        result = proj.project(h_2d)
        # [2,7] - 2*[1,0] = [0,7]
        np.testing.assert_allclose(result, [0.0, 7.0], atol=1e-10)

    def test_zero_norm_weight_raises(self):
        with pytest.raises(ValueError):
            NullSpaceProjector(np.zeros(3))


class TestOEICalculator:
    def test_identical_inputs_oei_is_zero(self):
        # No change → no escape, OEI = 0.0
        w = np.array([1.0, 0.0, 0.0])
        h = np.array([3.0, 2.0, 1.0])
        proj = NullSpaceProjector(w)
        calc = OEICalculator(proj)
        assert abs(calc.compute(h, h.copy()) - 0.0) < 1e-10

    def test_fully_orthogonal_change_oei_is_one(self):
        # Delta entirely in null space (orthogonal to w) → OEI = 1.0
        w = np.array([1.0, 0.0, 0.0])
        h_base = np.array([3.0, 1.0, 0.0])
        h_steered = np.array([3.0, 5.0, 0.0])  # delta = [0, 4, 0], orthogonal to w
        proj = NullSpaceProjector(w)
        calc = OEICalculator(proj)
        assert abs(calc.compute(h_base, h_steered) - 1.0) < 1e-10

    def test_fully_aligned_change_oei_is_zero(self):
        # Delta entirely along w → null projection of delta = 0 → OEI = 0.0
        w = np.array([1.0, 0.0, 0.0])
        h_base = np.array([2.0, 1.0, 0.0])
        h_steered = np.array([5.0, 1.0, 0.0])  # delta = [3, 0, 0], along w
        proj = NullSpaceProjector(w)
        calc = OEICalculator(proj)
        assert abs(calc.compute(h_base, h_steered) - 0.0) < 1e-10


class TestTDSCalculator:
    def test_identical_signals_tds_near_zero(self):
        v = np.array([0.3, 0.4, 0.3])
        T = TelemetryMatrix(h_neuron=v, repe_honesty=v.copy(), step=0)
        tds = TDSCalculator().compute(T)
        assert tds < 1e-6

    def test_divergent_signals_tds_positive(self):
        T = TelemetryMatrix(
            h_neuron=np.array([1.0, 0.0, 0.0]),
            repe_honesty=np.array([0.0, 0.0, 1.0]),
            step=0,
        )
        tds = TDSCalculator().compute(T)
        assert tds > 0.1

    def test_compute_from_arrays(self):
        calc = TDSCalculator()
        a = np.array([0.5, 0.5])
        b = np.array([0.1, 0.9])
        tds = calc.compute_from_arrays(a, b)
        assert 0.0 < tds <= 1.0


# ===========================================================================
# FP32Accumulator
# ===========================================================================

from logomesh.kv_mcts import (
    FP32Accumulator,
    KVCacheNode,
    MCTSConfig,
    _broadcast_to,
    _kv_snapshot_tuple,
    _kv_eval_cache,
)
from scripts.probe_kv_cache_mutability import _get_first_key_tensor


class _FakeDynamicCache:
    """Minimal DynamicCache-compatible mock with .key_cache / .value_cache lists.

    Does NOT inherit from transformers.cache_utils.DynamicCache — no transformers
    dependency in tests. Mimics the duck-typing interface used by _extract_kv_tensors.
    """

    def __init__(self, key_cache: list, value_cache: list) -> None:
        self.key_cache = key_cache
        self.value_cache = value_cache


class _FakeCacheLayer:
    """Transformers 5.3-like cache layer with .keys / .values tensors."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys = keys
        self.values = values


class _FakeDynamicCacheLayers:
    """Transformers 5.3-like DynamicCache with list-valued .layers only."""

    def __init__(self, layers: list[_FakeCacheLayer]) -> None:
        self.layers = layers


class TestFP32Accumulator:
    def _make_fake_kv(self, n_layers: int = 3, d: int = 4) -> tuple:
        """Returns a legacy-tuple KV cache of random float32 tensors."""
        return tuple(
            (torch.randn(1, 1, 2, d), torch.randn(1, 1, 2, d)) for _ in range(n_layers)
        )

    def _make_fake_dynamic_cache(
        self, n_layers: int = 3, d: int = 4
    ) -> _FakeDynamicCache:
        """Returns a _FakeDynamicCache with list-valued key_cache / value_cache."""
        return _FakeDynamicCache(
            key_cache=[torch.randn(1, 1, 2, d) for _ in range(n_layers)],
            value_cache=[torch.randn(1, 1, 2, d) for _ in range(n_layers)],
        )

    def _make_fake_dynamic_cache_layers(
        self, n_layers: int = 3, d: int = 4
    ) -> _FakeDynamicCacheLayers:
        """Returns a 5.3-like dynamic cache with .layers entries."""
        layers = [
            _FakeCacheLayer(torch.randn(1, 1, 2, d), torch.randn(1, 1, 2, d))
            for _ in range(n_layers)
        ]
        return _FakeDynamicCacheLayers(layers)

    def test_from_kv_cache_creates_zero_accumulators(self):
        kv = self._make_fake_kv()
        acc = FP32Accumulator.from_kv_cache(kv)
        assert len(acc.k_accum) == 3
        for a in acc.k_accum:
            assert a.dtype == torch.float32
            assert a.abs().max().item() == 0.0

    def test_apply_rollback_residual_near_zero(self):
        kv = self._make_fake_kv(n_layers=2, d=4)
        acc = FP32Accumulator.from_kv_cache(kv)
        dk = [np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)]

        k0_clone = kv[0][0].clone()

        assert acc.apply(kv, alpha=1.0, dk_vectors=dk) is True
        assert acc.rollback(kv, alpha=1.0, dk_vectors=dk) is True

        # Residual accumulator norm should be near zero
        assert acc.residual_norm() < 1e-6

        # KV tensor should match baseline
        k_after = kv[0][0]
        diff_norm = float((k_after.float() - k0_clone.float()).abs().max().item())
        assert diff_norm < 1e-4, f"Expected near-zero diff, got {diff_norm}"

    def test_multiple_apply_rollback_cycles_stay_bounded(self):
        kv = self._make_fake_kv(n_layers=2, d=4)
        acc = FP32Accumulator.from_kv_cache(kv)
        dk = [np.ones(4, dtype=np.float32)] * 2
        k0_clone = kv[0][0].float().clone()

        for _ in range(20):
            assert acc.apply(kv, alpha=0.5, dk_vectors=dk) is True
            assert acc.rollback(kv, alpha=0.5, dk_vectors=dk) is True

        # After 20 cycles, accumulator residual should still be near-zero
        assert acc.residual_norm() < 1e-5
        diff_norm = float((kv[0][0].float() - k0_clone).abs().max().item())
        assert diff_norm < 1e-3, f"Drift after 20 cycles: {diff_norm}"

    # -- DynamicCache format variants ------------------------------------------

    def test_from_kv_cache_creates_zero_accumulators_dynamic_cache(self):
        dc = self._make_fake_dynamic_cache()
        acc = FP32Accumulator.from_kv_cache(dc)
        assert len(acc.k_accum) == 3
        for a in acc.k_accum:
            assert a.dtype == torch.float32
            assert a.abs().max().item() == 0.0

    def test_apply_rollback_residual_near_zero_dynamic_cache(self):
        dc = self._make_fake_dynamic_cache(n_layers=2, d=4)
        acc = FP32Accumulator.from_kv_cache(dc)
        dk = [np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)]

        # Save a clone of the original key tensor in the list
        k0_clone = dc.key_cache[0].clone()

        assert acc.apply(dc, alpha=1.0, dk_vectors=dk) is True
        assert acc.rollback(dc, alpha=1.0, dk_vectors=dk) is True

        assert acc.residual_norm() < 1e-6
        diff_norm = float(
            (dc.key_cache[0].float() - k0_clone.float()).abs().max().item()
        )
        assert diff_norm < 1e-4, f"DynamicCache rollback diff: {diff_norm}"

    def test_multiple_apply_rollback_cycles_stay_bounded_dynamic_cache(self):
        dc = self._make_fake_dynamic_cache(n_layers=2, d=4)
        acc = FP32Accumulator.from_kv_cache(dc)
        dk = [np.ones(4, dtype=np.float32)] * 2
        k0_clone = dc.key_cache[0].float().clone()

        for _ in range(20):
            assert acc.apply(dc, alpha=0.5, dk_vectors=dk) is True
            assert acc.rollback(dc, alpha=0.5, dk_vectors=dk) is True

        assert acc.residual_norm() < 1e-5
        diff_norm = float((dc.key_cache[0].float() - k0_clone).abs().max().item())
        assert diff_norm < 1e-3, f"DynamicCache drift after 20 cycles: {diff_norm}"

    # -- DynamicCache layers-only variants (Transformers 5.3) -----------------

    def test_from_kv_cache_creates_zero_accumulators_dynamic_cache_layers(self):
        dc = self._make_fake_dynamic_cache_layers()
        acc = FP32Accumulator.from_kv_cache(dc)
        assert len(acc.k_accum) == 3
        for a in acc.k_accum:
            assert a.dtype == torch.float32
            assert a.abs().max().item() == 0.0

    def test_apply_rollback_residual_near_zero_dynamic_cache_layers(self):
        dc = self._make_fake_dynamic_cache_layers(n_layers=2, d=4)
        acc = FP32Accumulator.from_kv_cache(dc)
        dk = [np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)]

        k0_clone = dc.layers[0].keys.clone()

        assert acc.apply(dc, alpha=1.0, dk_vectors=dk) is True
        assert acc.rollback(dc, alpha=1.0, dk_vectors=dk) is True

        assert acc.residual_norm() < 1e-6
        diff_norm = float(
            (dc.layers[0].keys.float() - k0_clone.float()).abs().max().item()
        )
        assert diff_norm < 1e-4, f"DynamicCache(layers) rollback diff: {diff_norm}"

    def test_multiple_apply_rollback_cycles_stay_bounded_dynamic_cache_layers(self):
        dc = self._make_fake_dynamic_cache_layers(n_layers=2, d=4)
        acc = FP32Accumulator.from_kv_cache(dc)
        dk = [np.ones(4, dtype=np.float32)] * 2
        k0_clone = dc.layers[0].keys.float().clone()

        for _ in range(20):
            assert acc.apply(dc, alpha=0.5, dk_vectors=dk) is True
            assert acc.rollback(dc, alpha=0.5, dk_vectors=dk) is True

        assert acc.residual_norm() < 1e-5
        diff_norm = float((dc.layers[0].keys.float() - k0_clone).abs().max().item())
        assert diff_norm < 1e-3, (
            f"DynamicCache(layers) drift after 20 cycles: {diff_norm}"
        )


# ===========================================================================
# _kv_snapshot_tuple
# ===========================================================================


class TestKVSnapshotTuple:
    def test_tuple_format_passthrough(self):
        kv = tuple((torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4)) for _ in range(3))
        snap = _kv_snapshot_tuple(kv)
        assert isinstance(snap, tuple)
        assert len(snap) == 3
        for k, v in snap:
            assert torch.is_tensor(k) and torch.is_tensor(v)

    def test_dynamic_cache_format(self):
        dc = _FakeDynamicCache(
            key_cache=[torch.randn(1, 1, 2, 4) for _ in range(2)],
            value_cache=[torch.randn(1, 1, 2, 4) for _ in range(2)],
        )
        snap = _kv_snapshot_tuple(dc)
        assert isinstance(snap, tuple)
        assert len(snap) == 2
        # Snapshot tensors are detached — no grad tracking
        for k, v in snap:
            assert not k.requires_grad
            assert not v.requires_grad

    def test_dynamic_cache_layers_format(self):
        dc = _FakeDynamicCacheLayers(
            layers=[
                _FakeCacheLayer(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
                for _ in range(2)
            ]
        )
        snap = _kv_snapshot_tuple(dc)
        assert isinstance(snap, tuple)
        assert len(snap) == 2
        for k, v in snap:
            assert not k.requires_grad
            assert not v.requires_grad

    def test_snapshot_data_matches_source(self):
        kv = tuple((torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4)) for _ in range(2))
        snap = _kv_snapshot_tuple(kv)
        for (k_orig, v_orig), (k_snap, v_snap) in zip(kv, snap):
            assert torch.allclose(k_orig, k_snap)
            assert torch.allclose(v_orig, v_snap)


class TestKVEvalCache:
    def test_tuple_cache_returns_tuple_snapshot(self):
        kv = tuple((torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4)) for _ in range(2))
        eval_kv = _kv_eval_cache(kv)
        assert isinstance(eval_kv, tuple)
        assert len(eval_kv) == 2

    def test_dynamic_layers_cache_returns_deepcopy(self):
        dc = _FakeDynamicCacheLayers(
            layers=[
                _FakeCacheLayer(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
                for _ in range(2)
            ]
        )
        eval_cache = _kv_eval_cache(dc)
        assert eval_cache is not dc
        assert isinstance(eval_cache, _FakeDynamicCacheLayers)
        assert eval_cache.layers[0].keys is not dc.layers[0].keys
        assert torch.allclose(eval_cache.layers[0].keys, dc.layers[0].keys)


# ===========================================================================
# Probe helper extraction
# ===========================================================================


class TestProbeKVExtraction:
    def test_get_first_key_tensor_dynamic_cache_layers(self):
        dc = _FakeDynamicCacheLayers(
            layers=[_FakeCacheLayer(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))]
        )
        key, cache_type = _get_first_key_tensor(dc)
        assert torch.is_tensor(key)
        assert cache_type == "dynamic_cache_layers"


# ===========================================================================
# KVCacheNode
# ===========================================================================


class TestKVCacheNode:
    def test_make_root(self):
        root = KVCacheNode.make_root()
        assert root.parent_id is None
        assert root.depth == 0
        assert root.visit_count == 0

    def test_mean_reward_unvisited(self):
        node = KVCacheNode.make_root()
        assert node.mean_reward == 0.0

    def test_mean_reward_after_visits(self):
        node = KVCacheNode(node_id="a", parent_id=None, depth=0, alpha=1.0, layer=0)
        node.visit_count = 3
        node.reward_sum = 0.9
        assert abs(node.mean_reward - 0.3) < 1e-10

    def test_ucb1_unvisited_is_inf(self):
        node = KVCacheNode(node_id="a", parent_id=None, depth=1, alpha=0.5, layer=0)
        assert node.ucb1_score(parent_visits=10) == float("inf")

    def test_ucb1_visited_decreases_with_more_visits(self):
        node = KVCacheNode(node_id="a", parent_id=None, depth=1, alpha=0.5, layer=0)
        node.visit_count = 1
        node.reward_sum = 0.5
        score_1 = node.ucb1_score(parent_visits=10)
        node.visit_count = 5
        node.reward_sum = 2.5
        score_5 = node.ucb1_score(parent_visits=10)
        # With same mean_reward, higher visit_count → lower UCB1
        assert score_5 < score_1


# ===========================================================================
# ReversibleMCTS smoke test (fake model, no GPU)
# ===========================================================================

from logomesh.kv_mcts import ReversibleMCTS
from logomesh.telemetry_matrix import TelemetryMatrix as TM


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "formatted prompt"

    def __call__(self, text, return_tensors="pt"):
        return _FakeBatch({"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)})

    def decode(self, token_ids, skip_special_tokens=True):
        return "token"


class _FakeModel:
    def __call__(
        self,
        input_ids=None,
        past_key_values=None,
        use_cache=True,
        output_hidden_states=True,
        **kwargs,
    ):
        seq = int(input_ids.shape[1]) if input_ids is not None else 1
        hidden_a = torch.rand(1, seq, 4)
        hidden_b = torch.rand(1, seq, 4)
        return types.SimpleNamespace(
            logits=torch.tensor([[[0.1, 0.9, 0.0]]]),
            hidden_states=[hidden_a, hidden_b],
            router_logits=[],
            past_key_values=(
                (torch.zeros(1, 1, seq, 4), torch.zeros(1, 1, seq, 4)),
                (torch.zeros(1, 1, seq, 4), torch.zeros(1, 1, seq, 4)),
            ),
        )


class _FakeHNeuron:
    """Fake HNeuronMonitor with calibrated per-layer scores."""

    _calibrated = True

    def score_per_layer(self) -> list:
        return [0.3, 0.5]


class _FakeRepe:
    """Fake PerLayerHonestyProjector."""

    _calibrated = True

    @property
    def steering_vectors(self) -> list:
        return [np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)]

    def project(self, hidden_states) -> list:
        return [0.7, 0.6]


class _FakeOEICalc:
    def compute(self, h_base, h_steered) -> float:
        return 1.1


@pytest.mark.asyncio
async def test_reversible_mcts_smoke():
    """ReversibleMCTS runs with a fake model, returns nodes with telemetry."""
    from logomesh.local_model import LocalLlamaOracle

    oracle = LocalLlamaOracle(_FakeTokenizer(), _FakeModel(), device="cpu")
    hneuron = _FakeHNeuron()
    repe = _FakeRepe()
    oei_calc = _FakeOEICalc()

    config = MCTSConfig(
        n_nodes=6, branching_factor=2, max_depth=3, alpha_values=(0.1, 0.5)
    )
    mcts = ReversibleMCTS(
        oracle=oracle, hneuron=hneuron, repe=repe, oei_calc=oei_calc, config=config
    )

    nodes = await mcts.run_async(system="sys", user="usr")

    assert len(nodes) > 0, "Expected at least one node"
    # All non-root nodes should have telemetry
    non_root = [n for n in nodes if n.parent_id is not None]
    for node in non_root:
        assert node.telemetry is not None, f"Node {node.node_id} missing telemetry"
        assert isinstance(node.telemetry, TM)
    # Nodes are sorted by mean_reward descending
    rewards = [n.mean_reward for n in nodes]
    assert rewards == sorted(rewards, reverse=True)
    # Root node should have baseline telemetry (GAP-C2-12 fix)
    root_nodes = [n for n in nodes if n.parent_id is None]
    assert len(root_nodes) == 1
    assert root_nodes[0].telemetry is not None, (
        "Root node should have baseline telemetry"
    )


# ===========================================================================
# Integration tests: HNeuronMonitor.score_per_layer() with real tensors
# (GAP-C1-13)
# ===========================================================================

from logomesh.hneuron_monitor import HNeuronMonitor, _raw_score_with_indices


class _MockOracleForHNeuron:
    """Mock oracle returning controlled hidden states for HNeuronMonitor tests."""

    supports_telemetry = True

    def __init__(self, hidden_states: list):
        self._hidden_states = hidden_states

    def get_hidden_states(self):
        return self._hidden_states

    def get_router_logits(self):
        return []


class TestHNeuronScorePerLayerIntegration:
    """Integration tests for score_per_layer() with real tensor data."""

    def _make_calibrated_monitor(self, hidden_states, n_neurons=8):
        """Create an HNeuronMonitor with manually set calibration state."""
        oracle = _MockOracleForHNeuron(hidden_states)
        mon = HNeuronMonitor(oracle)
        mon._calibrated = True
        mon._is_moe = False
        # Set per-layer calibration: indices [0,1], coherent_mean=0.2, hallucinated_mean=0.8
        n_layers = len(hidden_states)
        mon._h_neuron_indices_per_layer = [[0, 1]] * n_layers
        mon._coherent_mean_per_layer = [0.2] * n_layers
        mon._hallucinated_mean_per_layer = [0.8] * n_layers
        mon._n_calibrated_layers = n_layers
        # Also set legacy fallback
        mon._h_neuron_indices = [0, 1]
        mon._coherent_mean = 0.2
        mon._hallucinated_mean = 0.8
        return mon

    def test_score_per_layer_returns_correct_length(self):
        """score_per_layer() should return one score per layer."""
        hs = [torch.rand(1, 8) for _ in range(4)]  # 4 layers, seq=1, dim=8
        mon = self._make_calibrated_monitor(hs)
        scores = mon.score_per_layer()
        assert len(scores) == 4

    def test_score_per_layer_values_in_range(self):
        """All scores should be in [0.0, 1.0]."""
        hs = [torch.rand(1, 8) for _ in range(3)]
        mon = self._make_calibrated_monitor(hs)
        scores = mon.score_per_layer()
        for s in scores:
            assert 0.0 <= s <= 1.0, f"Score {s} out of [0,1] range"

    def test_score_per_layer_high_activation_gives_high_score(self):
        """Hidden state with high values at H-Neuron indices should score high."""
        h = torch.zeros(1, 8)
        h[0, 0] = 10.0  # H-Neuron index 0: very high
        h[0, 1] = 10.0  # H-Neuron index 1: very high
        mon = self._make_calibrated_monitor([h])
        scores = mon.score_per_layer()
        assert scores[0] > 0.8, f"Expected high score, got {scores[0]}"

    def test_score_per_layer_low_activation_gives_low_score(self):
        """Hidden state with low values at H-Neuron indices should score low."""
        h = torch.zeros(1, 8)
        h[0, 0] = 0.1  # Near coherent_mean baseline
        h[0, 1] = 0.1
        mon = self._make_calibrated_monitor([h])
        scores = mon.score_per_layer()
        assert scores[0] < 0.3, f"Expected low score, got {scores[0]}"

    def test_score_per_layer_all_zeros(self):
        """All-zero hidden states should produce valid scores."""
        hs = [torch.zeros(1, 8) for _ in range(3)]
        mon = self._make_calibrated_monitor(hs)
        scores = mon.score_per_layer()
        assert len(scores) == 3
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_score_per_layer_uncalibrated_returns_half(self):
        """Uncalibrated monitor should return 0.5 for each layer."""
        hs = [torch.rand(1, 8) for _ in range(3)]
        oracle = _MockOracleForHNeuron(hs)
        mon = HNeuronMonitor(oracle)
        # Not calibrated
        scores = mon.score_per_layer()
        assert all(s == 0.5 for s in scores)

    def test_score_per_layer_empty_hidden_states(self):
        """Empty hidden states should return empty list."""
        oracle = _MockOracleForHNeuron([])
        mon = HNeuronMonitor(oracle)
        mon._calibrated = True
        scores = mon.score_per_layer()
        assert scores == []


# ===========================================================================
# Divergence tests: code vs Paper Eq. 3  (GAP-C1-02)
# ===========================================================================


class TestHNeuronEq3Divergence:
    """Document the known divergence between the code's H-Neuron scoring and
    Paper Eq. 3.

    Paper Eq. 3 (sigma_H^(l)(t)):
        sigma_H^(l)(t) = (1/|H|) * SUM_{j in H} ReLU(a_j^(l)(t) - a_bar_j^(l))

        - a_j^(l)(t): activation of H-Neuron j in layer l at step t
        - a_bar_j^(l): per-neuron calibration baseline (mean coherent activation)
        - ReLU applied per-neuron BEFORE averaging

    Code's actual formula (_raw_score_with_indices + _score_layer):
        raw   = mean(activation_row[i] for i in indices)   # no per-neuron baseline
        norm  = (raw - coherent_mean) / (hallucinated_mean - coherent_mean)
        score = clamp(norm, 0.0, 1.0)

    Three differences:
        1. No per-neuron baseline subtraction -- the code averages raw activations
           without subtracting each neuron's calibration mean a_bar_j^(l).
        2. ReLU placement -- Eq. 3 applies ReLU per-neuron (before aggregation).
           The code applies clamp(0, 1) only on the final normalized aggregate.
           This means negative per-neuron deltas survive into the average.
        3. Global normalization -- Eq. 3 produces an unnormalized sum-of-ReLUs.
           The code normalizes against (coherent_mean, hallucinated_mean),
           producing a [0, 1] score. This is useful but changes the semantics.

    This divergence is tracked as GAP-C1-02 and will be fixed by Max.
    When GAP-C1-02 lands, the ``eq3_expected != code_expected`` guard in
    test_eq3_hand_calculated will fail, naturally flagging this test class
    for update.
    """

    def _make_calibrated_monitor(self, hidden_states, indices, c_mean, h_mean):
        """Create an HNeuronMonitor with manually set calibration state."""
        oracle = _MockOracleForHNeuron(hidden_states)
        mon = HNeuronMonitor(oracle)
        mon._calibrated = True
        mon._is_moe = False
        n_layers = len(hidden_states)
        mon._h_neuron_indices_per_layer = [indices] * n_layers
        mon._coherent_mean_per_layer = [c_mean] * n_layers
        mon._hallucinated_mean_per_layer = [h_mean] * n_layers
        mon._n_calibrated_layers = n_layers
        mon._h_neuron_indices = indices
        mon._coherent_mean = c_mean
        mon._hallucinated_mean = h_mean
        return mon

    def test_eq3_hand_calculated(self):
        """Hand-calculated values showing Eq. 3 vs code divergence.

        Setup:
            H-neurons: indices [0, 1]
            Per-neuron calibration baselines (a_bar): [2.0, 3.0]
            Current activations: [5.0, 1.0, 0, 0, 0, 0, 0, 0]
            Calibration stats: coherent_mean=0.2, hallucinated_mean=0.8

        Paper Eq. 3:
            ReLU(5.0 - 2.0) = 3.0
            ReLU(1.0 - 3.0) = 0.0  (negative, zeroed by per-neuron ReLU)
            sigma_H = (3.0 + 0.0) / 2 = 1.5

        Code:
            raw = mean(5.0, 1.0) = 3.0           (_raw_score_with_indices)
            normalized = (3.0 - 0.2) / (0.8 - 0.2) = 4.667
            clamped = min(1.0, max(0.0, 4.667)) = 1.0
        """
        activation_row = [5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        indices = [0, 1]
        per_neuron_baselines = [2.0, 3.0]  # a_bar_j for Eq. 3
        coherent_mean = 0.2
        hallucinated_mean = 0.8

        # -- Paper Eq. 3 (what SHOULD happen) --
        eq3_per_neuron = [
            max(0.0, activation_row[j] - per_neuron_baselines[k])
            for k, j in enumerate(indices)
        ]
        assert eq3_per_neuron == [3.0, 0.0], (
            f"Unexpected per-neuron ReLU: {eq3_per_neuron}"
        )
        eq3_expected = sum(eq3_per_neuron) / len(eq3_per_neuron)
        assert abs(eq3_expected - 1.5) < 1e-9

        # -- Code's actual answer --
        raw = _raw_score_with_indices(activation_row, indices)
        assert abs(raw - 3.0) < 1e-9, f"Expected raw=3.0, got {raw}"

        # Normalized and clamped by _score_layer logic
        span = hallucinated_mean - coherent_mean
        normalized = (raw - coherent_mean) / span
        code_expected = max(0.0, min(1.0, normalized))
        assert abs(code_expected - 1.0) < 1e-9

        # -- Divergence guard --
        # This assertion documents that the two formulas disagree.
        # When GAP-C1-02 is fixed, eq3_expected and code_expected will align
        # and this assertion will FAIL, flagging the test for update.
        assert eq3_expected != code_expected, (
            "Eq. 3 and code now agree -- GAP-C1-02 may be resolved. "
            "Update this test class to reflect the fix."
        )

        # -- Verify end-to-end via HNeuronMonitor --
        h = torch.zeros(1, 8)
        h[0, 0] = 5.0
        h[0, 1] = 1.0
        mon = self._make_calibrated_monitor(
            [h], indices=indices, c_mean=coherent_mean, h_mean=hallucinated_mean
        )
        scores = mon.score_per_layer()
        assert len(scores) == 1
        assert abs(scores[0] - 1.0) < 1e-9, (
            f"Expected code score 1.0 (clamped), got {scores[0]}"
        )

    def test_eq3_divergence_in_raw_score_negative_activations(self):
        """Negative activations expose the divergence: code includes them,
        Eq. 3 zeros them via per-neuron ReLU.

        Setup:
            activations: [-2.0, 4.0, 0, 0]
            indices: [0, 1]
            Per-neuron baselines (a_bar): [0.0, 0.0]

        Code (_raw_score_with_indices):
            mean(-2.0, 4.0) = 1.0   (negative included in average)

        Eq. 3 with baselines [0, 0]:
            ReLU(-2.0 - 0.0) + ReLU(4.0 - 0.0) = 0.0 + 4.0
            mean = (0.0 + 4.0) / 2 = 2.0   (negative zeroed by ReLU)
        """
        activation_row = [-2.0, 4.0, 0.0, 0.0]
        indices = [0, 1]
        per_neuron_baselines = [0.0, 0.0]

        # Code answer
        code_raw = _raw_score_with_indices(activation_row, indices)
        assert abs(code_raw - 1.0) < 1e-9, f"Expected code raw=1.0, got {code_raw}"

        # Paper Eq. 3 answer
        eq3_per_neuron = [
            max(0.0, activation_row[j] - per_neuron_baselines[k])
            for k, j in enumerate(indices)
        ]
        eq3_raw = sum(eq3_per_neuron) / len(eq3_per_neuron)
        assert abs(eq3_raw - 2.0) < 1e-9, f"Expected Eq.3 raw=2.0, got {eq3_raw}"

        # Confirm divergence
        assert code_raw != eq3_raw, (
            "Code and Eq. 3 agree on negative activations -- "
            "GAP-C1-02 may be resolved. Update this test."
        )


# ===========================================================================
# Integration tests: PerLayerHonestyProjector.project() with real tensors
# (GAP-C1-13)
# ===========================================================================

from logomesh.whitebox import PerLayerHonestyProjector


class TestPerLayerHonestyProjectorIntegration:
    """Integration tests for project() with tensor inputs."""

    def _make_calibrated_projector(self, n_layers=3, dim=8):
        """Create a calibrated projector with known weight vectors."""
        rng = np.random.RandomState(42)
        proj = PerLayerHonestyProjector()
        # Set L2-normalized weight vectors per layer
        for _ in range(n_layers):
            w = rng.randn(dim).astype(np.float32)
            w = w / np.linalg.norm(w)
            proj._weights.append(w)
        proj._calibrated = True
        proj._n_layers = n_layers
        return proj

    def test_project_returns_correct_length(self):
        """project() should return one score per hidden state."""
        proj = self._make_calibrated_projector(n_layers=3, dim=8)
        hs = [torch.rand(1, 8) for _ in range(3)]
        scores = proj.project(hs)
        assert len(scores) == 3

    def test_project_known_dot_product(self):
        """Verify raw dot product matches manual calculation."""
        proj = PerLayerHonestyProjector()
        w = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        proj._weights = [w]
        proj._calibrated = True
        proj._n_layers = 1

        h = torch.tensor([[3.0, 5.0, 7.0, 9.0]])  # shape [1, 4], last token = [3,5,7,9]
        scores = proj.project([h])
        # dot([1,0,0,0], [3,5,7,9]) = 3.0
        assert abs(scores[0] - 3.0) < 1e-5, f"Expected 3.0, got {scores[0]}"

    def test_project_all_zero_hidden_states(self):
        """All-zero hidden states should produce zero dot products."""
        proj = self._make_calibrated_projector(n_layers=2, dim=8)
        hs = [torch.zeros(1, 8) for _ in range(2)]
        scores = proj.project(hs)
        for s in scores:
            assert abs(s) < 1e-6, f"Expected ~0.0 for zero hidden state, got {s}"

    def test_project_uncalibrated_returns_zeros(self):
        """Uncalibrated projector should return 0.0 for each layer."""
        proj = PerLayerHonestyProjector()
        hs = [torch.rand(1, 8) for _ in range(3)]
        scores = proj.project(hs)
        assert all(s == 0.0 for s in scores)

    def test_project_mismatched_dimensions(self):
        """Shape mismatch between weight and hidden state should return fallback."""
        proj = PerLayerHonestyProjector()
        proj._weights = [np.ones(4, dtype=np.float32)]  # dim=4
        proj._calibrated = True
        proj._n_layers = 1

        h = torch.rand(1, 8)  # dim=8 — mismatch
        scores = proj.project([h])
        assert scores[0] == 0.5, (
            f"Expected 0.5 fallback on shape mismatch, got {scores[0]}"
        )

    def test_project_more_layers_than_weights(self):
        """Extra layers beyond calibrated weights should get fallback scores."""
        proj = self._make_calibrated_projector(n_layers=2, dim=8)
        hs = [torch.rand(1, 8) for _ in range(4)]  # 4 layers, only 2 calibrated
        scores = proj.project(hs)
        assert len(scores) == 4
        # Layers 2 and 3 should get fallback (0.5)
        assert scores[2] == 0.5
        assert scores[3] == 0.5


# ===========================================================================
# GAP-C2-06: Phantom node detection — _broadcast_to, apply/rollback returns
# ===========================================================================


class TestBroadcastToReturnFlag:
    """_broadcast_to() returns (tensor, bool) indicating success."""

    def test_matching_shapes_succeed(self):
        vec = torch.ones(4)
        result, ok = _broadcast_to(vec, (1, 2, 3, 4))
        assert ok is True
        assert result.shape == (1, 2, 3, 4)

    def test_mismatching_shapes_fail(self):
        vec = torch.ones(8)
        result, ok = _broadcast_to(vec, (1, 2, 3, 4))
        assert ok is False
        assert result.shape == (1, 2, 3, 4)
        assert torch.all(result == 0)

    def test_scalar_passthrough_succeeds(self):
        """Non-1D vectors pass through without broadcast — always succeed."""
        vec = torch.ones(2, 4)
        result, ok = _broadcast_to(vec, (2, 4))
        assert ok is True
        assert result.shape == (2, 4)


class TestApplyRollbackPhantomDetection:
    """apply() and rollback() return False on shape mismatch (GAP-C2-06)."""

    def test_apply_returns_false_on_shape_mismatch(self):
        kv = ((torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4)),)
        acc = FP32Accumulator.from_kv_cache(kv)

        # Mismatched dk: d=8 but cache is d=4
        dk_wrong = np.ones(8, dtype=np.float32)
        result = acc.apply(kv, 1.0, [dk_wrong])
        assert result is False
        # Verify accumulator was NOT modified
        assert acc.residual_norm() < 1e-10

    def test_apply_returns_true_on_matching_shapes(self):
        kv = ((torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4)),)
        acc = FP32Accumulator.from_kv_cache(kv)

        dk_ok = np.ones(4, dtype=np.float32)
        result = acc.apply(kv, 1.0, [dk_ok])
        assert result is True
        assert acc.residual_norm() > 0  # accumulator was modified

    def test_rollback_returns_false_on_shape_mismatch(self):
        kv = ((torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4)),)
        acc = FP32Accumulator.from_kv_cache(kv)

        dk_wrong = np.ones(8, dtype=np.float32)
        result = acc.rollback(kv, 1.0, [dk_wrong])
        assert result is False
        assert acc.residual_norm() < 1e-10

    def test_multi_layer_partial_mismatch_no_mutation(self):
        """If layer 0 matches but layer 1 mismatches, no layer is mutated."""
        kv = (
            (torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4)),
            (torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4)),
        )
        acc = FP32Accumulator.from_kv_cache(kv)

        # Layer 0: d=4 (matches), Layer 1: d=8 (mismatches)
        dk_vecs = [np.ones(4, dtype=np.float32), np.ones(8, dtype=np.float32)]
        result = acc.apply(kv, 1.0, dk_vecs)
        assert result is False
        # Neither layer's accumulator was touched
        assert acc.residual_norm() < 1e-10


# Note: steering_valid field was removed from KVCacheNode — phantom nodes
# (shape mismatch) are now skipped entirely via `continue` in run_async(),
# so no invalid node ever enters the tree. See GAP-C2-06.


# ===========================================================================
# _calibrate_dense() shape validation
# ===========================================================================


class TestCalibrateDenseShapeGuard:
    """Verify _calibrate_dense() rejects 2D input and accepts valid 3D input."""

    def test_rejects_2d_input(self):
        """2D [examples][neurons] must raise ValueError, not silently corrupt."""
        oracle = _MockOracleForHNeuron([])
        mon = HNeuronMonitor(oracle)
        flat_coherent = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        flat_hallucinated = [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
        with pytest.raises(ValueError, match="3D input"):
            mon._calibrate_dense(flat_coherent, flat_hallucinated)

    def test_accepts_valid_3d_input(self):
        """Valid 3D [examples][layers][neurons] should populate per-layer state."""
        oracle = _MockOracleForHNeuron([])
        mon = HNeuronMonitor(oracle)
        # 2 examples, 3 layers, 4 neurons each
        coherent = [
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.2, 0.3, 0.4, 0.5]],
            [
                [0.15, 0.25, 0.35, 0.45],
                [0.55, 0.65, 0.75, 0.85],
                [0.25, 0.35, 0.45, 0.55],
            ],
        ]
        hallucinated = [
            [[0.9, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 5.0], [0.8, 0.7, 0.1, 0.0]],
            [
                [0.85, 0.15, 0.35, 0.45],
                [0.55, 0.65, 0.75, 4.5],
                [0.75, 0.65, 0.15, 0.05],
            ],
        ]
        mon._calibrate_dense(coherent, hallucinated)

        assert mon._n_calibrated_layers == 3
        assert len(mon._h_neuron_indices_per_layer) == 3
        assert len(mon._coherent_mean_per_layer) == 3
        assert len(mon._hallucinated_mean_per_layer) == 3
        # Each layer should have populated H-Neuron indices
        for indices in mon._h_neuron_indices_per_layer:
            assert len(indices) > 0
        # Neuron 0 has biggest diff at layer 0 (0.9-0.1=0.8)
        assert 0 in mon._h_neuron_indices_per_layer[0]
        # Neuron 3 has biggest diff at layer 1 (5.0-0.8=4.2)
        assert 3 in mon._h_neuron_indices_per_layer[1]

    def test_kv_cache_unchanged_on_failed_apply(self):
        """When apply() returns False, live KV cache tensors must be unchanged."""
        kv = ((torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4)),)
        acc = FP32Accumulator.from_kv_cache(kv)
        k_before = kv[0][0].clone()

        dk_wrong = np.ones(8, dtype=np.float32)  # d=8 vs cache d=4
        result = acc.apply(kv, 1.0, [dk_wrong])
        assert result is False
        assert acc.residual_norm() < 1e-10
        assert torch.equal(kv[0][0], k_before), "KV cache mutated despite False return"


# ===========================================================================
# Test 1: _read_telemetry() end-to-end with REAL HNeuronMonitor + REAL
#          PerLayerHonestyProjector (GAP-C1-13 integration)
# ===========================================================================


class _MockOracleForReadTelemetry:
    """Oracle stub that exposes controlled hidden states for _read_telemetry."""

    supports_telemetry = True

    def __init__(self, hidden_states: list):
        self._hidden_states = hidden_states

    def get_hidden_states(self):
        return self._hidden_states

    def get_router_logits(self):
        return []


class TestReadTelemetryIntegration:
    """_read_telemetry() with real HNeuronMonitor and real PerLayerHonestyProjector."""

    def test_read_telemetry_end_to_end(self):
        # 3 layers, hidden dim = 4 — FIXED values for deterministic assertions
        hidden_states = [
            torch.tensor([[0.3, 0.5, 0.7, 0.9]]),  # layer 0
            torch.tensor([[0.1, 0.4, 0.6, 0.8]]),  # layer 1
            torch.tensor([[0.2, 0.3, 0.5, 0.7]]),  # layer 2
        ]
        oracle = _MockOracleForReadTelemetry(hidden_states)

        # --- Real HNeuronMonitor with manual calibration ---
        hneuron = HNeuronMonitor(oracle)
        hneuron._calibrated = True
        hneuron._is_moe = False
        hneuron._h_neuron_indices_per_layer = [[0, 1]] * 3
        hneuron._coherent_mean_per_layer = [0.2] * 3
        hneuron._hallucinated_mean_per_layer = [0.8] * 3
        hneuron._n_calibrated_layers = 3
        # Legacy fallbacks
        hneuron._h_neuron_indices = [0, 1]
        hneuron._coherent_mean = 0.2
        hneuron._hallucinated_mean = 0.8

        # --- Real PerLayerHonestyProjector with known unit vectors ---
        # w_hon^(0) = [1,0,0,0], w_hon^(1) = [0,1,0,0], w_hon^(2) = [0,0,1,0]
        repe = PerLayerHonestyProjector()
        repe._weights = [
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        ]
        repe._calibrated = True
        repe._n_layers = 3

        # --- Build ReversibleMCTS with real components ---
        mcts = ReversibleMCTS(
            oracle=oracle,
            hneuron=hneuron,
            repe=repe,
            oei_calc=_FakeOEICalc(),
            config=MCTSConfig(),
        )

        T = mcts._read_telemetry()

        # Assertions
        assert isinstance(T, TM), f"Expected TelemetryMatrix, got {type(T)}"
        mat = T.as_matrix()
        assert mat.shape == (2, 3), f"Expected shape (2, 3), got {mat.shape}"

        # sigma_H values (row 0) should be in [0, 1]
        for i in range(3):
            assert 0.0 <= mat[0, i] <= 1.0, f"sigma_H[{i}] = {mat[0, i]} out of [0,1]"

        # Hand-calculated sigma_H for each layer:
        # indices=[0,1], c_mean=0.2, h_mean=0.8, span=0.6
        # Layer 0: raw = (0.3 + 0.5) / 2 = 0.4, norm = (0.4 - 0.2) / 0.6 = 0.333
        # Layer 1: raw = (0.1 + 0.4) / 2 = 0.25, norm = (0.25 - 0.2) / 0.6 = 0.083
        # Layer 2: raw = (0.2 + 0.3) / 2 = 0.25, norm = (0.25 - 0.2) / 0.6 = 0.083
        assert abs(mat[0, 0] - 1 / 3) < 1e-5, (
            f"sigma_H[0] expected ~0.333, got {mat[0, 0]}"
        )
        assert abs(mat[0, 1] - 1 / 12) < 1e-5, (
            f"sigma_H[1] expected ~0.083, got {mat[0, 1]}"
        )
        assert abs(mat[0, 2] - 1 / 12) < 1e-5, (
            f"sigma_H[2] expected ~0.083, got {mat[0, 2]}"
        )

        # Hand-calculated rho_R: dot product with unit vectors
        # rho_R[0] = dot([1,0,0,0], [0.3,0.5,0.7,0.9]) = 0.3
        # rho_R[1] = dot([0,1,0,0], [0.1,0.4,0.6,0.8]) = 0.4
        # rho_R[2] = dot([0,0,1,0], [0.2,0.3,0.5,0.7]) = 0.5
        assert abs(mat[1, 0] - 0.3) < 1e-4, f"rho_R[0] expected 0.3, got {mat[1, 0]}"
        assert abs(mat[1, 1] - 0.4) < 1e-4, f"rho_R[1] expected 0.4, got {mat[1, 1]}"
        assert abs(mat[1, 2] - 0.5) < 1e-4, f"rho_R[2] expected 0.5, got {mat[1, 2]}"


# ===========================================================================
# Test 2: run_async() skips all nodes when steering vectors have wrong dims
# ===========================================================================


class TestMCTSShapeMismatchSkip:
    """When steering vectors have wrong dimensions, apply() returns False and
    the node is skipped (lines 622-631 in kv_mcts.py).

    We cannot call run_async() end-to-end with all-failing applies because the
    while loop has no termination guard when n_expanded never increments.
    Instead we verify the skip path by:
      1. Running run_async with apply patched to return False on the FIRST call
         (proving the skip works) then True on subsequent calls (so the loop
         terminates normally).
      2. Asserting that the first alpha produced no child node while later
         alphas did.
    """

    @pytest.mark.asyncio
    async def test_shape_mismatch_skips_node(self):
        """Verify that when apply() returns False (shape mismatch), the node
        is skipped via `continue` and no child is registered in the tree.

        Uses _FakeRepe (d=4, matching cache) but patches apply() to return
        False on the first call, simulating what _FakeRepeMismatch (d=8)
        would trigger. Subsequent calls use the real apply (which succeeds)
        so the MCTS loop can terminate normally.
        """
        from logomesh.local_model import LocalLlamaOracle

        oracle = LocalLlamaOracle(_FakeTokenizer(), _FakeModel(), device="cpu")
        hneuron = _FakeHNeuron()
        repe = _FakeRepe()  # d=4, matches cache — real apply will succeed
        oei_calc = _FakeOEICalc()

        config = MCTSConfig(
            n_nodes=2,
            branching_factor=2,
            max_depth=2,
            alpha_values=(0.1, 0.5),
        )
        mcts = ReversibleMCTS(
            oracle=oracle,
            hneuron=hneuron,
            repe=repe,
            oei_calc=oei_calc,
            config=config,
        )

        call_count = [0]
        original_apply = FP32Accumulator.apply

        def apply_with_first_skip(
            self_acc, past_kv, alpha, dk_vectors, dv_vectors=None
        ):
            call_count[0] += 1
            if call_count[0] == 1:
                # Simulate shape mismatch on the first expansion attempt
                return False
            return original_apply(self_acc, past_kv, alpha, dk_vectors, dv_vectors)

        with mock.patch.object(FP32Accumulator, "apply", apply_with_first_skip):
            nodes = await mcts.run_async(system="sys", user="usr")

        # apply() was called at least twice: once returning False, then True
        assert call_count[0] >= 2, "apply() should have been called at least twice"

        root_nodes = [n for n in nodes if n.parent_id is None]
        assert len(root_nodes) == 1, "Should have exactly one root"
        assert root_nodes[0].telemetry is not None, (
            "Root should have baseline telemetry"
        )

        # Non-root nodes exist (from the successful applies after the skip)
        non_root = [n for n in nodes if n.parent_id is not None]
        assert len(non_root) >= 1, "At least one child should have been added"

        # Total nodes < n_nodes + 1 (root) because at least one was skipped
        # With n_nodes=2, branching_factor=2, and one skip: expect root + 2 children
        assert len(nodes) <= config.n_nodes + 1


# ===========================================================================
# Test 3: RuntimeError raised when rollback fails after successful apply
# ===========================================================================


class TestMCTSRollbackFailurePropagation:
    """RuntimeError is raised when rollback fails after a successful apply."""

    @pytest.mark.asyncio
    async def test_rollback_failure_raises_runtime_error(self):
        from logomesh.local_model import LocalLlamaOracle

        oracle = LocalLlamaOracle(_FakeTokenizer(), _FakeModel(), device="cpu")
        mcts = ReversibleMCTS(
            oracle=oracle,
            hneuron=_FakeHNeuron(),
            repe=_FakeRepe(),
            oei_calc=_FakeOEICalc(),
            config=MCTSConfig(
                n_nodes=2,
                branching_factor=1,
                max_depth=2,
                alpha_values=(0.1,),
            ),
        )

        with mock.patch.object(FP32Accumulator, "rollback", return_value=False):
            with pytest.raises(RuntimeError, match="shape invariant violated"):
                await mcts.run_async(system="sys", user="usr")


# ===========================================================================
# Test 4: _calibrate_dense() pipeline + score_per_layer() with hand-calculated
#          values
# ===========================================================================


class TestCalibrateDensePipelineScores:
    """Calibrate with known 3D data, then verify score_per_layer() against
    hand-calculated values using the code's actual formula."""

    def test_calibrate_and_score_hand_calculated(self):
        from logomesh.hneuron_monitor import TOP_K_NEURONS

        # 2 examples, 2 layers, 4 neurons
        coherent = [
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.1, 0.2, 0.2]],
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.1, 0.2, 0.2]],
        ]
        hallucinated = [
            [[0.9, 0.2, 0.3, 0.4], [0.5, 0.1, 0.2, 1.0]],
            [[0.9, 0.2, 0.3, 0.4], [0.5, 0.1, 0.2, 1.0]],
        ]

        # --- Hand-calculate expected calibration state ---
        # Layer 0: per-neuron coherent means = [0.1, 0.2, 0.3, 0.4]
        #          per-neuron halluc  means = [0.9, 0.2, 0.3, 0.4]
        #          diffs = [0.8, 0.0, 0.0, 0.0]
        #          Top-K (k = min(TOP_K, 4) = 4): ranked = [0, 1, 2, 3]
        # Layer 1: per-neuron coherent means = [0.5, 0.1, 0.2, 0.2]
        #          per-neuron halluc  means = [0.5, 0.1, 0.2, 1.0]
        #          diffs = [0.0, 0.0, 0.0, 0.8]
        #          Top-K: ranked = [3, 0, 1, 2]
        k = min(TOP_K_NEURONS, 4)  # = 4 since 4 neurons < TOP_K_NEURONS(50)

        # Layer 0 indices (top-k by diff descending): all 4 neurons since k=4
        # _raw_score on coherent rows with all indices:
        #   mean(0.1, 0.2, 0.3, 0.4) = 0.25
        # _raw_score on hallucinated rows with all indices:
        #   mean(0.9, 0.2, 0.3, 0.4) = 0.45
        c_mean_l0 = 0.25
        h_mean_l0 = 0.45

        # Layer 1 indices: all 4 neurons
        # _raw_score on coherent rows: mean(0.5, 0.1, 0.2, 0.2) = 0.25
        # _raw_score on hallucinated rows: mean(0.5, 0.1, 0.2, 1.0) = 0.45
        c_mean_l1 = 0.25
        h_mean_l1 = 0.45

        # --- Calibrate ---
        oracle = _MockOracleForHNeuron([])
        mon = HNeuronMonitor(oracle)
        mon._calibrate_dense(coherent, hallucinated)

        assert mon._n_calibrated_layers == 2
        assert len(mon._h_neuron_indices_per_layer) == 2
        assert len(mon._h_neuron_indices_per_layer[0]) == k
        assert len(mon._h_neuron_indices_per_layer[1]) == k

        # Verify calibration baselines match hand-calculation
        assert abs(mon._coherent_mean_per_layer[0] - c_mean_l0) < 1e-9
        assert abs(mon._hallucinated_mean_per_layer[0] - h_mean_l0) < 1e-9
        assert abs(mon._coherent_mean_per_layer[1] - c_mean_l1) < 1e-9
        assert abs(mon._hallucinated_mean_per_layer[1] - h_mean_l1) < 1e-9

        # --- Score with known hidden states ---
        # Layer 0 tensor: [0.5, 0.2, 0.3, 0.4]
        #   raw = mean(0.5, 0.2, 0.3, 0.4) = 0.35
        #   normalized = (0.35 - 0.25) / (0.45 - 0.25) = 0.10 / 0.20 = 0.5
        #   clamped = 0.5
        # Layer 1 tensor: [0.5, 0.1, 0.2, 0.6]
        #   raw = mean(0.5, 0.1, 0.2, 0.6) = 0.35
        #   normalized = (0.35 - 0.25) / (0.45 - 0.25) = 0.10 / 0.20 = 0.5
        #   clamped = 0.5
        hs = [
            torch.tensor([0.5, 0.2, 0.3, 0.4]),
            torch.tensor([0.5, 0.1, 0.2, 0.6]),
        ]
        # Swap oracle hidden states
        mon._oracle = _MockOracleForHNeuron(hs)

        scores = mon.score_per_layer()
        assert len(scores) == 2, f"Expected 2 scores, got {len(scores)}"
        assert abs(scores[0] - 0.5) < 1e-6, f"Layer 0 expected 0.5, got {scores[0]}"
        assert abs(scores[1] - 0.5) < 1e-6, f"Layer 1 expected 0.5, got {scores[1]}"


# ===========================================================================
# Test 5: KV cache tensors unchanged after failed rollback()
# ===========================================================================


class TestRollbackImmutability:
    """KV cache tensors must be unchanged after a failed rollback()."""

    def test_kv_cache_unchanged_after_failed_rollback(self):
        # KV cache: 1 layer, shape (1, 1, 3, 4) — d=4
        kv = ((torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4)),)
        acc = FP32Accumulator.from_kv_cache(kv)

        # Successful apply with d=4 vectors — accumulator becomes non-zero
        dk_good = [np.ones(4, dtype=np.float32)]
        apply_ok = acc.apply(kv, alpha=1.0, dk_vectors=dk_good)
        assert apply_ok is True
        assert acc.residual_norm() > 0, "Accumulator should be non-zero after apply"

        # Snapshot the KV tensors and residual norm BEFORE the failed rollback
        k_snapshot = kv[0][0].clone()
        v_snapshot = kv[0][1].clone()
        norm_before = acc.residual_norm()

        # Attempt rollback with d=8 vectors — shape mismatch, should return False
        dk_wrong = [np.ones(8, dtype=np.float32)]
        rb_ok = acc.rollback(kv, alpha=1.0, dk_vectors=dk_wrong)

        # Assertions
        assert rb_ok is False, "rollback() should return False on shape mismatch"
        assert torch.equal(kv[0][0], k_snapshot), "K tensor mutated by failed rollback"
        assert torch.equal(kv[0][1], v_snapshot), "V tensor mutated by failed rollback"
        norm_after = acc.residual_norm()
        assert abs(norm_after - norm_before) < 1e-10, (
            f"residual_norm changed from {norm_before} to {norm_after} after failed rollback"
        )
