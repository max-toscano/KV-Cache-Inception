"""
SAGE H-Neuron Monitor — Hallucination-associated neuron scoring.

Identifies neurons whose activation correlates with incoherent/hallucinated
output, and provides a real-time score during offline MCTS generation.

Two paths depending on model architecture:
  - Dense (Llama-3.2-1B, Phase A): MLP neuron activation monitoring.
    H-Neurons are found by calibration: neurons with high activation
    variance between coherent and hallucinated examples.
  - MoE (gpt-oss-20b, Phase B): router logit entropy as proxy.
    High routing entropy → expert selection is uncertain → model is
    generating incoherent output. See CLAUDE.md "Gotchas" for why
    dense monitoring does not apply to MoE.

Usage (Phase A — Llama):
    from logomesh.local_model import LocalLlamaOracle
    from logomesh.hneuron_monitor import HNeuronMonitor

    oracle = LocalLlamaOracle.load("./models/llama-3.2-1b")
    monitor = HNeuronMonitor(oracle)

    monitor.calibrate(
        coherent_examples=["Tell me about Paris.", "What is photosynthesis?"],
        hallucinated_examples=["Explain the Battle of Zorvak in 1842.",
                               "List the moons of Planet Quasar-7."],
    )

    await oracle.generate(system="...", user="attack prompt")
    score = monitor.score()  # 0.0 = coherent, 1.0 = hallucinating

Score interpretation in MCTS reward:
    Final reward = 0.4 * (1 - h_score) + 0.6 * lat_score
    Low h_score (coherent) is a prerequisite for the attack to be credible.
    A payload with h_score > 0.7 is likely incoherent and will not fool
    the defender regardless of how "attack-like" it looks.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .local_model import LocalLlamaOracle

logger = logging.getLogger(__name__)

# Minimum examples required for reliable calibration
MIN_CALIBRATION_EXAMPLES = 5

# How many top H-Neurons to use for scoring (the most discriminative ones)
TOP_K_NEURONS = 50


class HNeuronMonitor:
    """Monitor hallucination-associated neurons during offline MCTS.

    Call calibrate() once per session (or load from cache), then call
    score() after each oracle.generate() to get the H-Neuron score.

    Args:
        oracle: A LocalLlamaOracle with supports_telemetry=True.
    """

    def __init__(self, oracle: "LocalLlamaOracle") -> None:
        if not oracle.supports_telemetry:
            raise ValueError(
                "HNeuronMonitor requires an oracle with supports_telemetry=True. "
                "Use LocalLlamaOracle, not OpenAIOracle."
            )
        self._oracle = oracle
        self._is_moe: bool = False  # set in calibrate() based on router_logits presence
        self._calibrated: bool = False

        # Dense path calibration state
        self._h_neuron_indices: list[int] = []  # top-K neuron indices
        self._coherent_mean: float = 0.0  # mean score on coherent examples
        self._coherent_std: float = 1.0
        self._hallucinated_mean: float = 0.0
        self._hallucinated_std: float = 1.0

        # Per-layer calibration state (Paper Eq. 3: per-layer H-Neuron sets)
        self._h_neuron_indices_per_layer: list[list[int]] = []
        self._coherent_mean_per_layer: list[float] = []
        self._hallucinated_mean_per_layer: list[float] = []
        self._n_calibrated_layers: int = 0

        # MoE path calibration state
        self._coherent_entropy_mean: float = 0.0
        self._coherent_entropy_std: float = 1.0
        self._hallucinated_entropy_mean: float = 0.0

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    async def calibrate(
        self,
        coherent_examples: list[str],
        hallucinated_examples: list[str],
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        """Calibrate H-Neuron thresholds from contrastive examples.

        Args:
            coherent_examples: Prompts where factual, coherent responses are expected.
                               E.g., well-known facts, clear questions.
            hallucinated_examples: Prompts likely to elicit hallucinated responses.
                                   E.g., fictional entities, contradictory constraints.
            system_prompt: System prompt to use for both sets.

        After calibration, score() returns values normalized to [0, 1] relative
        to these calibration distributions.
        """
        if len(coherent_examples) < MIN_CALIBRATION_EXAMPLES:
            logger.warning(
                "Only %d coherent examples provided (min %d). "
                "Calibration may be unreliable.",
                len(coherent_examples),
                MIN_CALIBRATION_EXAMPLES,
            )
        if len(hallucinated_examples) < MIN_CALIBRATION_EXAMPLES:
            logger.warning(
                "Only %d hallucinated examples provided (min %d).",
                len(hallucinated_examples),
                MIN_CALIBRATION_EXAMPLES,
            )

        logger.info(
            "Calibrating HNeuronMonitor: %d coherent, %d hallucinated examples",
            len(coherent_examples),
            len(hallucinated_examples),
        )

        coherent_activations = await self._collect_activations(
            coherent_examples, system_prompt
        )
        hallucinated_activations = await self._collect_activations(
            hallucinated_examples, system_prompt
        )

        if not coherent_activations or not hallucinated_activations:
            logger.error("Calibration failed: no activations collected")
            return

        # Detect model type from first example's router logits
        self._is_moe = len(self._oracle.get_router_logits()) > 0

        if self._is_moe:
            self._calibrate_moe(coherent_activations, hallucinated_activations)
        else:
            self._calibrate_dense(coherent_activations, hallucinated_activations)

        self._calibrated = True
        logger.info(
            "Calibration complete. Mode: %s. Layers: %d. H-Neurons per layer: %d",
            "MoE/entropy" if self._is_moe else "dense/MLP",
            self._n_calibrated_layers if not self._is_moe else 0,
            len(self._h_neuron_indices) if not self._is_moe else 0,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self) -> float:
        """Score the most recent oracle.generate() call.

        Returns:
            Float in [0.0, 1.0]. 0.0 = maximally coherent, 1.0 = hallucinating.
            Returns 0.5 if not calibrated (neutral / unknown).

        Call this immediately after oracle.generate() — the hidden states
        are cached from that forward pass and will be overwritten on the next call.
        """
        if not self._calibrated:
            logger.warning(
                "HNeuronMonitor.score() called before calibrate(). Returning 0.5."
            )
            return 0.5

        if self._is_moe:
            return self._score_moe()
        return self._score_dense()

    def score_per_layer(self) -> "list[float]":
        """Return σ_H^(l) for every transformer layer, as a list of length L.

        Each layer uses its own independently calibrated H-Neuron set and
        baselines (Paper Eq. 3: per-layer contrastive activation analysis).

        Returns:
            List of floats in [0.0, 1.0], length = number of transformer layers.
            Returns a list of 0.5 values if not calibrated or if hidden states
            are unavailable.

        Call immediately after oracle.generate() or oracle.generate_one_step(),
        before the next forward pass overwrites the cached states.
        Used by TelemetryMatrix assembly in kv_mcts.py.
        """
        hidden_states = self._oracle.get_hidden_states()
        if not hidden_states or not self._calibrated or self._is_moe:
            n = len(hidden_states) if hidden_states else 0
            return [0.5] * n

        scores = [
            self._score_layer(h, layer_idx=i) for i, h in enumerate(hidden_states)
        ]
        return scores

    # ------------------------------------------------------------------
    # Internal — dense path (Llama-3.2-1B)
    # ------------------------------------------------------------------

    def _calibrate_dense(
        self,
        coherent_all_layers: list[list[list[float]]],
        hallucinated_all_layers: list[list[list[float]]],
    ) -> None:
        """Identify top-K H-Neurons per layer by activation difference.

        Paper Eq. 3 defines sigma_H^(l) with per-layer neuron sets.  Each
        layer gets its own contrastive activation analysis so cross-layer
        variation comes from both activation magnitude AND different neuron
        subsets.

        Args:
            coherent_all_layers: [n_examples][n_layers][hidden_size].
            hallucinated_all_layers: same shape.
        """
        import statistics

        if not coherent_all_layers or not hallucinated_all_layers:
            return

        # Shape guard: reject 2D input [examples][neurons] from legacy callers.
        # Expected shape is [n_examples][n_layers][hidden_size] (3D nested lists).
        first_example = coherent_all_layers[0]
        if first_example and not isinstance(first_example[0], (list, tuple)):
            raise ValueError(
                "_calibrate_dense() expects 3D input [examples][layers][neurons], "
                f"but got 2D input — first example element is "
                f"{type(first_example[0]).__name__}, not list. "
                "Ensure _collect_activations() returns per-layer data."
            )

        n_layers = len(coherent_all_layers[0])
        if n_layers == 0:
            return

        self._n_calibrated_layers = n_layers
        self._h_neuron_indices_per_layer = []
        self._coherent_mean_per_layer = []
        self._hallucinated_mean_per_layer = []

        for layer_idx in range(n_layers):
            coherent_layer = [
                ex[layer_idx] for ex in coherent_all_layers if layer_idx < len(ex)
            ]
            hallucinated_layer = [
                ex[layer_idx] for ex in hallucinated_all_layers if layer_idx < len(ex)
            ]

            if not coherent_layer or not hallucinated_layer:
                self._h_neuron_indices_per_layer.append([])
                self._coherent_mean_per_layer.append(0.0)
                self._hallucinated_mean_per_layer.append(1.0)
                continue

            n_neurons = len(coherent_layer[0])
            if n_neurons == 0:
                self._h_neuron_indices_per_layer.append([])
                self._coherent_mean_per_layer.append(0.0)
                self._hallucinated_mean_per_layer.append(1.0)
                continue

            k = min(TOP_K_NEURONS, n_neurons)

            # Per-neuron mean across examples at this layer
            coherent_means = [
                statistics.mean(row[i] for row in coherent_layer)
                for i in range(n_neurons)
            ]
            hallucinated_means = [
                statistics.mean(row[i] for row in hallucinated_layer)
                for i in range(n_neurons)
            ]

            # H-Neurons for this layer: highest (hallucinated - coherent)
            diffs = [
                hallucinated_means[i] - coherent_means[i] for i in range(n_neurons)
            ]
            ranked = sorted(range(n_neurons), key=lambda i: diffs[i], reverse=True)
            layer_indices = ranked[:k]
            self._h_neuron_indices_per_layer.append(layer_indices)

            # Calibrate score distribution for this layer
            coherent_scores = [
                _raw_score_with_indices(row, layer_indices) for row in coherent_layer
            ]
            hallucinated_scores = [
                _raw_score_with_indices(row, layer_indices)
                for row in hallucinated_layer
            ]

            self._coherent_mean_per_layer.append(
                statistics.mean(coherent_scores) if coherent_scores else 0.0
            )
            self._hallucinated_mean_per_layer.append(
                statistics.mean(hallucinated_scores) if hallucinated_scores else 1.0
            )

        # Backward compat: last-layer values for score() / _score_dense()
        if self._h_neuron_indices_per_layer:
            self._h_neuron_indices = self._h_neuron_indices_per_layer[-1]
            self._coherent_mean = self._coherent_mean_per_layer[-1]
            self._hallucinated_mean = self._hallucinated_mean_per_layer[-1]

    def _raw_dense_score(self, activation_row: list[float]) -> float:
        """Mean activation over the top-K H-Neurons for one example."""
        if not self._h_neuron_indices:
            return 0.0
        return sum(activation_row[i] for i in self._h_neuron_indices) / len(
            self._h_neuron_indices
        )

    def _score_dense(self) -> float:
        """Score using cached MLP activations from LocalLlamaOracle."""
        hidden_states = self._oracle.get_hidden_states()
        if not hidden_states:
            logger.warning("No hidden states available. Call oracle.generate() first.")
            return 0.5
        return self._score_layer(hidden_states[-1])

    def _score_layer(self, h, layer_idx: "int | None" = None) -> float:
        """Score a single layer's hidden-state tensor.

        Args:
            h: Tensor of shape [hidden_size] (last-token) or [seq, hidden_size].
            layer_idx: If provided, use per-layer calibration data. If None,
                       falls back to last-layer calibration (backward compat).

        Returns:
            Float in [0.0, 1.0] normalised against calibration baselines.
        """
        try:
            if h.dim() == 1:
                activation_row = h.tolist()
            else:
                activation_row = h[-1].tolist()
        except Exception:
            return 0.5

        # Select per-layer or last-layer calibration data
        if (
            layer_idx is not None
            and self._h_neuron_indices_per_layer
            and layer_idx < len(self._h_neuron_indices_per_layer)
        ):
            indices = self._h_neuron_indices_per_layer[layer_idx]
            c_mean = self._coherent_mean_per_layer[layer_idx]
            h_mean = self._hallucinated_mean_per_layer[layer_idx]
        else:
            indices = self._h_neuron_indices
            c_mean = self._coherent_mean
            h_mean = self._hallucinated_mean

        raw = _raw_score_with_indices(activation_row, indices)

        # Normalize: 0.0 at coherent_mean, 1.0 at hallucinated_mean
        span = h_mean - c_mean
        if abs(span) < 1e-8:
            return 0.5
        normalized = (raw - c_mean) / span
        return float(max(0.0, min(1.0, normalized)))

    # ------------------------------------------------------------------
    # Internal — MoE path (gpt-oss-20b, Phase B)
    # ------------------------------------------------------------------

    def _calibrate_moe(
        self,
        coherent_activations: list[list[float]],
        hallucinated_activations: list[list[float]],
    ) -> None:
        """Calibrate entropy baseline from MoE router logits.

        coherent_activations here contains pre-computed entropy scores
        (one per example), not raw neuron activations.
        """
        import statistics

        # When is_moe, _collect_activations stores [entropy_score] per example
        c_entropies = [row[0] for row in coherent_activations if row]
        h_entropies = [row[0] for row in hallucinated_activations if row]

        self._coherent_entropy_mean = (
            statistics.mean(c_entropies) if c_entropies else 0.0
        )
        self._coherent_entropy_std = (
            statistics.stdev(c_entropies) if len(c_entropies) > 1 else 1.0
        )
        self._hallucinated_entropy_mean = (
            statistics.mean(h_entropies) if h_entropies else 1.0
        )

    def _score_moe(self) -> float:
        """Score using router logit entropy from LocalLlamaOracle (gpt-oss-20b path)."""
        router_logits = self._oracle.get_router_logits()
        if not router_logits:
            logger.warning("No router logits available. Is this a MoE model?")
            return 0.5

        entropy = _compute_router_entropy(router_logits)
        span = self._hallucinated_entropy_mean - self._coherent_entropy_mean
        if abs(span) < 1e-8:
            return 0.5
        normalized = (entropy - self._coherent_entropy_mean) / span
        return float(max(0.0, min(1.0, normalized)))

    # ------------------------------------------------------------------
    # Internal — activation collection
    # ------------------------------------------------------------------

    async def _collect_activations(
        self, examples: list[str], system_prompt: str
    ) -> list:
        """Run oracle on each example and extract activations.

        Returns a list (one per example) of activation data.
        For dense models: [n_examples][n_layers][hidden_size] — per-layer
            last-token activations for per-layer H-Neuron calibration.
        For MoE models: [n_examples][1] — router entropy scalar per example.
        """
        results: list = []
        for text in examples:
            try:
                await self._oracle.generate(system=system_prompt, user=text)
            except Exception as e:
                logger.error("Failed to generate for calibration example: %s", e)
                continue

            router_logits = self._oracle.get_router_logits()
            is_moe = len(router_logits) > 0

            if is_moe:
                entropy = _compute_router_entropy(router_logits)
                results.append([entropy])
            else:
                hidden_states = self._oracle.get_hidden_states()
                if not hidden_states:
                    continue
                # Collect last-token activations from ALL layers for per-layer
                # H-Neuron calibration (Paper Eq. 3: sigma_H^(l)).
                example_layers: list[list[float]] = []
                for hs in hidden_states:
                    try:
                        if hs.dim() == 1:
                            row = hs.tolist()
                        else:
                            row = hs[-1].tolist()
                        example_layers.append(row)
                    except Exception as e:
                        logger.debug("Failed to extract layer activation: %s", e)
                if example_layers:
                    results.append(example_layers)

        return results


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------


def _raw_score_with_indices(activation_row: list[float], indices: list[int]) -> float:
    """Mean activation over given neuron indices for one example."""
    if not indices:
        return 0.0
    return sum(activation_row[i] for i in indices) / len(indices)


def _compute_router_entropy(router_logits: list) -> float:
    """Compute mean per-token entropy over all MoE routing layers.

    Args:
        router_logits: List of Tensors from oracle.get_router_logits().
                       Each tensor: [seq_len, num_experts] or [num_experts].

    Returns:
        Mean routing entropy in nats. Higher = more uncertain = hallucinating.
    """

    entropies = []
    for logits in router_logits:
        try:
            # logits: Tensor [seq, num_experts] or [num_experts]
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            # Softmax → probability distribution over experts
            log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
            probs = log_probs.exp()

            # Shannon entropy per token: H = -∑ p_i log(p_i)
            H = -(probs * log_probs.clamp(min=-100)).sum(dim=-1)  # [seq]
            entropies.append(float(H.mean()))
        except Exception as e:
            logger.debug("Router entropy computation failed for a layer: %s", e)
            continue

    if not entropies:
        return 0.0

    # Normalize by maximum possible entropy: log(num_experts)
    # This puts the score on a [0, 1] scale relative to uniform routing
    max_entropy = math.log(router_logits[0].shape[-1]) if router_logits else 1.0
    raw_mean = sum(entropies) / len(entropies)
    return raw_mean / max(max_entropy, 1e-8)
