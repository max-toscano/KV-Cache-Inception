"""
LogoMesh Reversible MCTS in KV-Cache Latent Space.

Implements the paper's second core contribution: Monte Carlo Tree Search
operating directly in the model's KV-cache latent space using mathematically
reversible FP32 accumulator rollbacks (Theorem 1).

Key classes:
  FP32Accumulator  — tracks cumulative steering deltas in FP32, enables exact
                     reversal independent of rollback count (Theorem 1)
  KVCacheNode      — MCTS tree node: state, telemetry, UCB1 scoring
  MCTSConfig       — hyperparameters (branching, depth, reward weights)
  ReversibleMCTS   — full MCTS loop: selection, expansion, evaluation, backprop

Equations implemented:
  Forward mutation (paper §4.2):
    K_t^(l) ← K_t^(l) + α·d_K^(l),  V_t^(l) ← V_t^(l) + α·d_V^(l)

  FP32 accumulator (Eq. 6):
    A^(l) ← A^(l) + α·d^(l)  [FP32],  K_t^(l) ← K_base^(l) + cast(A^(l))

  UCB1 (Eq. ucb1):
    UCB1(i) = r̄_i + c·√(2 ln N / n_i)

  Node reward (Eq. 8):
    r(node) = λ₁·σ̄_H + λ₂·(1−ρ̄_R) + λ₃·TDS(σ_H, ρ_R)

Usage:
    from logomesh.kv_mcts import ReversibleMCTS, MCTSConfig
    from logomesh.hneuron_monitor import HNeuronMonitor
    from logomesh.whitebox import PerLayerHonestyProjector
    from logomesh.orthogonal_escape import OEICalculator, NullSpaceProjector

    config = MCTSConfig(n_nodes=50, branching_factor=3, max_depth=5)
    mcts = ReversibleMCTS(oracle, hneuron, repe, oei_calc, config)
    nodes = mcts.run(system="...", user="...")
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .hneuron_monitor import HNeuronMonitor
    from .local_model import LocalLlamaOracle
    from .orthogonal_escape import OEICalculator
    from .telemetry_matrix import TelemetryMatrix
    from .whitebox import PerLayerHonestyProjector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------


def _get_transformers_version() -> str:
    try:
        import transformers

        return transformers.__version__
    except Exception:
        return "unknown"


def _extract_kv_from_layers(layers: Any) -> list[tuple[Any, Any]]:
    """Extract (K, V) tensors from cache layer objects.

    Supports layer entries as:
      - tuple/list: (K, V, ...)
      - object with .keys and .values (Transformers 5.3 CacheLayerMixin)
      - object with .key and .value (older/custom variants)

    Returns an empty list if schema is unrecognized.
    """
    if not isinstance(layers, (list, tuple)) or not layers:
        return []

    out: list[tuple[Any, Any]] = []
    for layer in layers:
        k = v = None

        if isinstance(layer, (tuple, list)) and len(layer) >= 2:
            k, v = layer[0], layer[1]
        elif hasattr(layer, "keys") and hasattr(layer, "values"):
            k, v = layer.keys, layer.values
        elif hasattr(layer, "key") and hasattr(layer, "value"):
            k, v = layer.key, layer.value

        # Some cache layers may be lazily initialised.
        if k is None or v is None:
            return []

        out.append((k, v))

    return out


def _extract_kv_tensors(past_key_values: Any) -> list[tuple[Any, Any]]:
    """Return list of (K_tensor, V_tensor) per layer from past_key_values.

    Supports legacy tuple format, DynamicCache key_cache/value_cache format,
    and DynamicCache layers format used in Transformers 5.3.
    """
    if isinstance(past_key_values, (tuple, list)):
        return [(layer[0], layer[1]) for layer in past_key_values]

    # Explicit DynamicCache check (Transformers 5.x live formats)
    try:
        from transformers.cache_utils import DynamicCache

        if isinstance(past_key_values, DynamicCache):
            # Transformers <=5.2 style: list-valued key_cache/value_cache
            if hasattr(past_key_values, "key_cache") and hasattr(
                past_key_values, "value_cache"
            ):
                kc = past_key_values.key_cache
                vc = past_key_values.value_cache
                if isinstance(kc, (list, tuple)) and isinstance(vc, (list, tuple)):
                    return list(zip(kc, vc))

            # Transformers 5.3 style: list-valued layers; each layer has keys/values
            if hasattr(past_key_values, "layers"):
                kv = _extract_kv_from_layers(past_key_values.layers)
                if kv:
                    return kv

            dyn_keys = sorted(list(getattr(past_key_values, "__dict__", {}).keys()))
            raise TypeError(
                "Unsupported DynamicCache schema. Expected either "
                "list-valued key_cache/value_cache or list-valued layers with "
                "keys/values tensors. "
                f"DynamicCache attrs: {dyn_keys}. "
                f"Installed transformers: {_get_transformers_version()}"
            )
    except ImportError:
        pass

    # Duck-typing fallback: any Cache subclass whose .key_cache / .value_cache are lists
    if hasattr(past_key_values, "key_cache") and hasattr(
        past_key_values, "value_cache"
    ):
        kc = past_key_values.key_cache
        vc = past_key_values.value_cache
        if isinstance(kc, (list, tuple)) and isinstance(vc, (list, tuple)):
            return list(zip(kc, vc))

    # Duck-typing fallback for layers-based caches.
    if hasattr(past_key_values, "layers"):
        kv = _extract_kv_from_layers(past_key_values.layers)
        if kv:
            return kv

    raise TypeError(
        f"Unsupported past_key_values type: {type(past_key_values)!r}. "
        f"Expected a tuple/list of (K, V) pairs or a DynamicCache subclass with "
        f"list-valued .key_cache / .value_cache, or a layers-based cache with "
        f"per-layer keys/values tensors. "
        f"Installed transformers: {_get_transformers_version()}"
    )


def _clone_kv_cache(past_key_values: Any) -> Any:
    """Deep-clone past_key_values preserving type."""
    import copy
    import torch

    if isinstance(past_key_values, (tuple, list)):
        return tuple(
            tuple(t.clone() if torch.is_tensor(t) else t for t in layer)
            for layer in past_key_values
        )
    return copy.deepcopy(past_key_values)


def _kv_snapshot_tuple(past_key_values: Any) -> tuple:
    """Extract current KV state as a detached legacy tuple — (K, V) per layer.

    Used for branch-evaluation forward passes to prevent DynamicCache.update()
    from replacing list entries with grown tensors during model execution.
    Uses .detach() without .clone() — no extra VRAM allocated, same storage.

    This is critical for Theorem 1 correctness: FP32Accumulator.rollback() requires
    that the tensors it holds references to in key_cache[l] are not replaced by the
    model between apply() and rollback(). Passing a tuple snapshot instead of the
    live DynamicCache prevents .update() from being called.
    """
    return tuple(
        (k.detach(), v.detach()) for k, v in _extract_kv_tensors(past_key_values)
    )


def _kv_eval_cache(past_key_values: Any) -> Any:
    """Return an evaluation-safe cache snapshot for one-step forward passes.

    Why this exists:
        - For tuple/list caches, a detached tuple view is sufficient and cheap.
        - For DynamicCache-style objects, many model forwards expect cache methods
            like get_seq_length(). Passing a tuple can fail. In those cases we pass
            a deep-copied cache object so model-side update() calls do not mutate
            the live cache tracked by FP32Accumulator.
    """
    if isinstance(past_key_values, (tuple, list)):
        return _kv_snapshot_tuple(past_key_values)
    return _clone_kv_cache(past_key_values)


# ---------------------------------------------------------------------------
# FP32 Accumulator — Theorem 1 exact reversibility
# ---------------------------------------------------------------------------


@dataclass
class FP32Accumulator:
    """Per-layer FP32 accumulators for exact KV-cache reversal (Theorem 1).

    Stores baseline KV tensors (K_base, V_base) and cumulative FP32 deltas
    (k_accum, v_accum) so that the cache can be reconstructed from:
        K_t^(l) = K_base^(l) + cast_to_model_dtype(k_accum^(l))

    After n complete apply/rollback cycles, residual_norm() ≤ ε_bf16 · ‖A_final‖_∞
    (independent of n), vs. naive subtraction bound of n · ε_bf16 · max ‖δ_i‖_∞.

    Attributes:
        k_base: Baseline K tensors, one per layer (cloned at root).
        v_base: Baseline V tensors, one per layer.
        k_accum: Cumulative K delta in FP32, same shape as k_base per layer.
        v_accum: Cumulative V delta in FP32.
    """

    k_base: list[Any]  # baseline K per layer (original model dtype)
    v_base: list[Any]  # baseline V per layer
    k_accum: list[Any]  # A_K^(l) in FP32
    v_accum: list[Any]  # A_V^(l) in FP32

    @classmethod
    def from_kv_cache(cls, past_key_values: Any) -> "FP32Accumulator":
        """Initialise accumulators from the root KV cache.

        Clones the baseline tensors and creates zero FP32 accumulators.

        Args:
            past_key_values: Output of a model forward pass (use_cache=True).

        Returns:
            FP32Accumulator ready for apply/rollback cycles.
        """
        import torch

        layers = _extract_kv_tensors(past_key_values)
        k_base, v_base, k_accum, v_accum = [], [], [], []
        for k, v in layers:
            k_base.append(k.clone())
            v_base.append(v.clone())
            k_accum.append(torch.zeros_like(k, dtype=torch.float32))
            v_accum.append(torch.zeros_like(v, dtype=torch.float32))
        return cls(k_base=k_base, v_base=v_base, k_accum=k_accum, v_accum=v_accum)

    def apply(
        self,
        past_key_values: Any,
        alpha: float,
        dk_vectors: list[np.ndarray],
        dv_vectors: list[np.ndarray] | None = None,
    ) -> bool:
        """Apply steering: A^(l) += α·d^(l), K^(l) = K_base^(l) + cast(A^(l)).
        Modifies past_key_values in-place.

        Uses two-pass pre-validation: all broadcasts are validated before any
        accumulator is mutated.  If any layer's broadcast fails, the method
        returns False and no state is changed.

        Args:
            past_key_values: Live KV cache to mutate.
            alpha: Steering magnitude.
            dk_vectors: Per-layer K steering directions d_K^(l), as np.ndarray [d_k].
            dv_vectors: Per-layer V steering directions (same as dk if None).

        Returns:
            True if steering was applied, False if a shape mismatch prevented it.
        """
        import torch

        if dv_vectors is None:
            dv_vectors = dk_vectors

        layers = _extract_kv_tensors(past_key_values)
        n = min(len(layers), len(dk_vectors), len(self.k_accum))

        # Pass 1: pre-validate all broadcasts (no mutations)
        broadcast_results: list[tuple[Any, Any]] = []
        for l_idx in range(n):
            k_live, v_live = layers[l_idx]
            dk = torch.from_numpy(dk_vectors[l_idx]).to(
                device=k_live.device, dtype=torch.float32
            )
            dv = torch.from_numpy(dv_vectors[l_idx]).to(
                device=v_live.device, dtype=torch.float32
            )

            dk, dk_ok = _broadcast_to(dk, k_live.shape)
            dv, dv_ok = _broadcast_to(dv, v_live.shape)

            if not dk_ok or not dv_ok:
                return False

            broadcast_results.append((dk, dv))

        # Pass 2: all broadcasts valid — apply mutations
        for l_idx in range(n):
            k_live, v_live = layers[l_idx]
            dk, dv = broadcast_results[l_idx]

            self.k_accum[l_idx].add_(alpha * dk)
            self.v_accum[l_idx].add_(alpha * dv)

            # Reconstruct: K = K_base + cast(A) [Eq. 6]
            k_live.copy_(
                self.k_base[l_idx]
                + self.k_accum[l_idx].to(dtype=self.k_base[l_idx].dtype)
            )
            v_live.copy_(
                self.v_base[l_idx]
                + self.v_accum[l_idx].to(dtype=self.v_base[l_idx].dtype)
            )

        return True

    def rollback(
        self,
        past_key_values: Any,
        alpha: float,
        dk_vectors: list[np.ndarray],
        dv_vectors: list[np.ndarray] | None = None,
    ) -> bool:
        """Reverse steering: A^(l) -= α·d^(l), K^(l) = K_base^(l) + cast(A^(l)).

        Uses two-pass pre-validation: all broadcasts are validated before any
        accumulator is mutated.  If any layer's broadcast fails, the method
        returns False and no state is changed.

        Args: same as apply().

        Returns:
            True if rollback was applied, False if a shape mismatch prevented it.
        """
        import torch

        if dv_vectors is None:
            dv_vectors = dk_vectors

        layers = _extract_kv_tensors(past_key_values)
        n = min(len(layers), len(dk_vectors), len(self.k_accum))

        # Pass 1: pre-validate all broadcasts (no mutations)
        broadcast_results: list[tuple[Any, Any]] = []
        for l_idx in range(n):
            k_live, v_live = layers[l_idx]
            dk = torch.from_numpy(dk_vectors[l_idx]).to(
                device=k_live.device, dtype=torch.float32
            )
            dv = torch.from_numpy(dv_vectors[l_idx]).to(
                device=v_live.device, dtype=torch.float32
            )

            dk, dk_ok = _broadcast_to(dk, k_live.shape)
            dv, dv_ok = _broadcast_to(dv, v_live.shape)

            if not dk_ok or not dv_ok:
                return False

            broadcast_results.append((dk, dv))

        # Pass 2: all broadcasts valid — apply rollback mutations
        for l_idx in range(n):
            k_live, v_live = layers[l_idx]
            dk, dv = broadcast_results[l_idx]

            self.k_accum[l_idx].sub_(alpha * dk)
            self.v_accum[l_idx].sub_(alpha * dv)

            k_live.copy_(
                self.k_base[l_idx]
                + self.k_accum[l_idx].to(dtype=self.k_base[l_idx].dtype)
            )
            v_live.copy_(
                self.v_base[l_idx]
                + self.v_accum[l_idx].to(dtype=self.v_base[l_idx].dtype)
            )

        return True

    def residual_norm(self) -> float:
        """‖A_K‖_∞ over all layers — should be ~0 after complete rollback.

        Theorem 1: ‖K_n − K_0‖_∞ ≤ ε_bf16 · ‖A_final‖_∞
        After a full apply+rollback cycle, A_final → 0 in FP32, so the
        bound collapses to near-machine-epsilon of bf16 (~3.9e-3).
        """
        if not self.k_accum:
            return 0.0
        return float(max(a.abs().max().item() for a in self.k_accum))


def _broadcast_to(vec, target_shape) -> tuple[Any, bool]:
    """Broadcast a 1-D steering vector to a target KV tensor shape.

    KV cache tensors have shape [batch, heads, seq, d_head] or [batch, seq, d_head].
    The steering vector has shape [d_model] or [d_head].  We expand it along
    all leading dimensions so the add_ is valid.

    Returns:
        (broadcast_tensor, success): The broadcast tensor and whether the
        broadcast succeeded.  On shape mismatch the tensor is zeros and
        success is False.
    """
    import torch

    if vec.dim() == 1 and len(target_shape) > 1:
        # Align the last dimension; expand all others
        tail_dim = target_shape[-1]
        if vec.shape[0] == tail_dim:
            for _ in range(len(target_shape) - 1):
                vec = vec.unsqueeze(0)
            vec = vec.expand(target_shape)
        else:
            # Shape mismatch — return zeros of correct shape (safe no-op)
            logger.warning(
                "_broadcast_to: shape mismatch — vec last dim %d != target last dim %d. "
                "Steering delta zeroed; this node's telemetry is invalid.",
                vec.shape[0],
                target_shape[-1],
            )
            return torch.zeros(target_shape, dtype=vec.dtype, device=vec.device), False
    return vec, True


# ---------------------------------------------------------------------------
# KVCacheNode — MCTS tree node
# ---------------------------------------------------------------------------


@dataclass
class KVCacheNode:
    """A node in the Reversible MCTS search tree.

    Attributes:
        node_id:     Unique identifier (UUID string).
        parent_id:   Parent node_id, or None for root.
        depth:       Depth in tree (root = 0).
        alpha:       Steering magnitude α applied at this node.
        layer:       Transformer layer index that was steered.
        visit_count: Number of times this node has been selected.
        reward_sum:  Cumulative reward accumulated over visits.
        children:    List of child node_ids.
        telemetry:   TelemetryMatrix T_t recorded at this node (set post-eval).
        oei_score:   OEI computed at this node (set post-eval).
    """

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

    @classmethod
    def make_root(cls) -> "KVCacheNode":
        return cls(
            node_id=str(uuid.uuid4()), parent_id=None, depth=0, alpha=0.0, layer=-1
        )

    @property
    def mean_reward(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.reward_sum / self.visit_count

    def ucb1_score(self, parent_visits: int, c: float = 1.414) -> float:
        """UCB1(i) = r̄_i + c·√(2 ln N / n_i).

        Unvisited nodes return +∞ (always expanded first).
        """
        if self.visit_count == 0:
            return float("inf")
        return self.mean_reward + c * math.sqrt(
            2.0 * math.log(max(parent_visits, 1)) / self.visit_count
        )


# ---------------------------------------------------------------------------
# MCTS configuration
# ---------------------------------------------------------------------------


@dataclass
class MCTSConfig:
    """Hyperparameters for ReversibleMCTS.

    Attributes:
        n_nodes:              Total nodes to expand (budget).
        branching_factor:     Children per expansion step.
        max_depth:            Maximum tree depth.
        alpha_values:         Steering magnitudes to try per expansion.
                              Maps to Experiment 1: α ∈ {0.1, 0.5, 1.0, 2.0, 5.0}.
        exploration_constant: UCB1 c = √2 by default.
        reward_lambdas:       (λ₁, λ₂, λ₃) for Eq. 8 node reward.
        steer_layer:          Transformer layer to steer. -1 = auto (middle layer).
    """

    n_nodes: int = 50
    branching_factor: int = 3
    max_depth: int = 5
    alpha_values: tuple = (0.1, 0.5, 1.0, 2.0, 5.0)
    exploration_constant: float = 1.414
    reward_lambdas: tuple = (0.33, 0.33, 0.34)
    steer_layer: int = -1


# ---------------------------------------------------------------------------
# ReversibleMCTS — main algorithm
# ---------------------------------------------------------------------------


class ReversibleMCTS:
    """Reversible Monte Carlo Tree Search in KV-Cache Latent Space.

    Each expansion:
      1. Selects an alpha from config.alpha_values
      2. Applies steering to the KV cache via FP32Accumulator.apply()
      3. Runs oracle.generate_one_step() to advance generation
      4. Reads T_t (telemetry matrix) from HNeuronMonitor + PerLayerHonestyProjector
      5. Computes node reward r(node) = Eq. 8
      6. Rolls back the KV cache via FP32Accumulator.rollback()
      7. Updates UCB1 statistics via backpropagation

    Args:
        oracle:   LocalLlamaOracle with generate_one_step() and telemetry accessors.
        hneuron:  HNeuronMonitor (must be calibrated before run()).
        repe:     PerLayerHonestyProjector (must be calibrated before run()).
        oei_calc: OEICalculator (used to record OEI per node).
        config:   MCTSConfig hyperparameters.
    """

    def __init__(
        self,
        oracle: "LocalLlamaOracle",
        hneuron: "HNeuronMonitor",
        repe: "PerLayerHonestyProjector",
        oei_calc: "OEICalculator",
        config: MCTSConfig | None = None,
    ) -> None:
        self._oracle = oracle
        self._hneuron = hneuron
        self._repe = repe
        self._oei_calc = oei_calc
        self._config = config or MCTSConfig()
        self._nodes: dict[str, KVCacheNode] = {}

    def run(self, system: str, user: str) -> list[KVCacheNode]:
        """Run MCTS and return all visited nodes sorted by mean_reward desc.

        This is an async method wrapped for sync call. Use run_async() from
        async contexts.

        Args:
            system: System prompt for the oracle.
            user:   User prompt to seed the initial KV cache.

        Returns:
            List of KVCacheNode, sorted by mean_reward descending.
        """
        return asyncio.get_event_loop().run_until_complete(self.run_async(system, user))

    async def run_async(self, system: str, user: str) -> list[KVCacheNode]:
        """Async MCTS run.

        Returns all visited nodes sorted by mean_reward descending.
        """
        from logomesh.telemetry_matrix import compute_node_reward

        # Seed the KV cache with one forward step on the prompt
        step0 = await self._oracle.generate_one_step(system=system, user=user)
        past_kv = step0["past_key_values"]

        if past_kv is None:
            logger.error("generate_one_step() returned no KV cache. Cannot run MCTS.")
            return []

        # Determine steering layer
        steer_layer = self._config.steer_layer
        if steer_layer < 0:
            n_layers = len(_extract_kv_tensors(past_kv))
            steer_layer = max(0, n_layers // 2)

        # Build FP32 accumulator from root KV cache
        accumulator = FP32Accumulator.from_kv_cache(past_kv)

        # Read initial hidden state for OEI baseline
        initial_hs = self._oracle.get_hidden_states()
        h_base = (
            initial_hs[steer_layer].float().detach().cpu().numpy()
            if initial_hs and steer_layer < len(initial_hs)
            else None
        )

        # Steering vectors from RepE projector (honesty direction = d_K^(l))
        steering_vecs = self._repe.steering_vectors
        if not steering_vecs:
            logger.warning(
                "PerLayerHonestyProjector has no steering vectors. Run calibrate() first."
            )
            steering_vecs = [np.zeros(1)]

        # Root node — evaluate baseline telemetry before any steering
        root = KVCacheNode.make_root()
        root.telemetry = self._read_telemetry()
        self._nodes[root.node_id] = root

        n_expanded = 0
        while n_expanded < self._config.n_nodes:
            # Selection: UCB1 among leaf-eligible nodes
            parent = self._select(root)
            if parent.depth >= self._config.max_depth:
                parent.visit_count += 1  # mark visited to reduce UCB1 pressure
                continue

            # Expansion: try each alpha as a child
            alphas_to_try = list(self._config.alpha_values)[
                : self._config.branching_factor
            ]
            for alpha in alphas_to_try:
                if n_expanded >= self._config.n_nodes:
                    break

                child_id = str(uuid.uuid4())
                child = KVCacheNode(
                    node_id=child_id,
                    parent_id=parent.node_id,
                    depth=parent.depth + 1,
                    alpha=alpha,
                    layer=steer_layer,
                )

                # Apply steering to KV cache (in-place, FP32 accumulator)
                dk = steering_vecs[min(steer_layer, len(steering_vecs) - 1)]
                steering_ok = accumulator.apply(
                    past_kv, alpha, [dk] * len(accumulator.k_accum)
                )
                if not steering_ok:
                    logger.warning(
                        "Steering shape mismatch at layer %d, alpha=%.2f — skipping node.",
                        steer_layer,
                        alpha,
                    )
                    continue  # No rollback needed — apply() made no mutations

                # One-step forward pass with mutated KV cache.
                # Pass a detached tuple snapshot, not the live DynamicCache, so
                # DynamicCache.update() cannot replace key_cache[l] entries with
                # grown tensors — which would break rollback() shape invariants.
                try:
                    step = await self._oracle.generate_one_step(
                        input_ids=step0["input_ids"],
                        past_key_values=_kv_eval_cache(past_kv),
                        temperature=0.0,
                    )
                except Exception as e:
                    logger.warning("MCTS generate_one_step failed: %s", e)
                    rb_ok = accumulator.rollback(
                        past_kv, alpha, [dk] * len(accumulator.k_accum)
                    )
                    if not rb_ok:
                        raise RuntimeError(
                            f"Rollback failed after successful apply at depth={parent.depth + 1}, "
                            f"alpha={alpha} — FP32 accumulator shape invariant violated."
                        )
                    continue

                # Read telemetry T_t
                T = self._read_telemetry()
                child.telemetry = T

                # OEI (if baseline hidden state available)
                if h_base is not None:
                    hs_now = self._oracle.get_hidden_states()
                    if hs_now and steer_layer < len(hs_now):
                        h_steered = hs_now[steer_layer].float().detach().cpu().numpy()
                        try:
                            child.oei_score = self._oei_calc.compute(h_base, h_steered)
                        except Exception as exc:
                            logger.warning(
                                "OEI computation failed for node %s: %s",
                                child_id,
                                exc,
                                exc_info=True,
                            )

                # Reward
                reward = compute_node_reward(T, self._config.reward_lambdas)

                # Rollback (Theorem 1 reversibility)
                rb_ok = accumulator.rollback(
                    past_kv, alpha, [dk] * len(accumulator.k_accum)
                )
                if not rb_ok:
                    raise RuntimeError(
                        f"Rollback failed after successful apply at depth={parent.depth + 1}, "
                        f"alpha={alpha} — FP32 accumulator shape invariant violated."
                    )

                # Register child and backpropagate
                parent.children.append(child_id)
                self._nodes[child_id] = child
                self._backpropagate(child, reward)

                n_expanded += 1

        logger.info(
            "MCTS complete: %d nodes expanded, steer_layer=%d, residual_norm=%.2e",
            n_expanded,
            steer_layer,
            accumulator.residual_norm(),
        )

        return sorted(self._nodes.values(), key=lambda n: n.mean_reward, reverse=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_telemetry(self) -> "TelemetryMatrix":
        """Assemble T_t from HNeuronMonitor and PerLayerHonestyProjector."""
        from logomesh.telemetry_matrix import TelemetryMatrix

        sigma_H = self._hneuron.score_per_layer()
        hs = self._oracle.get_hidden_states()
        rho_R = self._repe.project(hs) if hs else ([0.0] * len(sigma_H))

        n = max(len(sigma_H), len(rho_R))
        if len(sigma_H) < n:
            sigma_H = list(sigma_H) + [0.5] * (n - len(sigma_H))
        if len(rho_R) < n:
            rho_R = list(rho_R) + [0.0] * (n - len(rho_R))

        return TelemetryMatrix(
            h_neuron=np.array(sigma_H, dtype=np.float32),
            repe_honesty=np.array(rho_R, dtype=np.float32),
            step=0,
        )

    def _select(self, root: KVCacheNode) -> KVCacheNode:
        """UCB1 selection: traverse tree from root to best leaf."""
        node = root
        while node.children:
            best_child = max(
                (self._nodes[cid] for cid in node.children),
                key=lambda c: c.ucb1_score(
                    node.visit_count, self._config.exploration_constant
                ),
            )
            if best_child.visit_count == 0:
                return best_child
            node = best_child
        return node

    def _backpropagate(self, node: KVCacheNode, reward: float) -> None:
        """Update visit_count and reward_sum from node back to root."""
        current: KVCacheNode | None = node
        while current is not None:
            current.visit_count += 1
            current.reward_sum += reward
            pid = current.parent_id
            current = self._nodes.get(pid) if pid else None


# Touch marker: 2026-04-14 commit-date-check.
