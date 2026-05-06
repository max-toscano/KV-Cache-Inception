"""
LogoMesh LocalLlamaOracle — HuggingFace transformers wrapper for offline MCTS (Phase A).

Implements BaseModelClient and exposes hidden states and router logits for
H-Neuron monitoring and LAT probe training. Extended in Phase 2 to support
single-step generation and mutable KV-cache access for Reversible MCTS.

Phase A: Llama-3.2-1B-Instruct (dense, ~2GB VRAM on RTX 3060)
Phase B: gpt-oss-20b (MoE, ~16GB+ VRAM, H100 only)
         Requires: pip install git+https://github.com/huggingface/transformers
         (GptOssForCausalLM is not in stable transformers releases)

Usage:
    from logomesh.local_model import LocalLlamaOracle

    oracle = LocalLlamaOracle.load("./models/llama-3.2-1b")
    await oracle.generate(system="...", user="...")
    hidden = oracle.get_hidden_states()   # list[Tensor], one per layer
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

try:
    from jinja2 import TemplateError as _Jinja2TemplateError
except ImportError:  # jinja2 not installed (unlikely — transitive dep of transformers)
    _Jinja2TemplateError = Exception  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Exception types raised by tokenizer.apply_chat_template() on template issues.
_TEMPLATE_ERRORS = (ValueError, TypeError, KeyError, _Jinja2TemplateError)


def _resolve_model_ref(model_path: str) -> str:
    """Return the model reference to pass to from_pretrained.

    If model_path points to an existing local directory, return it as-is.
    Otherwise treat it as a HuggingFace model id (hub handles download/cache).
    """
    path = Path(model_path)
    if path.exists():
        return str(path)
    # Treat as HuggingFace model id — do not require local existence
    logger.info(
        "Local path '%s' not found — treating as HuggingFace model id (will use hub cache).",
        model_path,
    )
    return model_path


def _load_model(model_path: str, device: str):
    """Load tokenizer + model from a local directory or HuggingFace model id.

    Separated from __init__ so callers can control when the heavy import
    and CUDA allocation happen (e.g. only in offline scripts, not tests).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_ref = _resolve_model_ref(model_path)

    logger.info("Loading tokenizer from %s", model_ref)
    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if device != "cpu" else torch.float32
    logger.info("Loading model from %s onto %s (dtype=%s)", model_ref, device, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    logger.info("Model loaded: %s params", sum(p.numel() for p in model.parameters()))
    return tokenizer, model


class LocalLlamaOracle:
    """BaseModelClient backed by a local HuggingFace transformers model.

    Implements the BaseModelClient interface plus telemetry accessors used by
    HNeuronMonitor and LAT probe training.

    The model is loaded lazily via LocalLlamaOracle.load() rather than
    in __init__, so importing this module in tests does not trigger CUDA
    allocation.

    Example
    -------
    oracle = LocalLlamaOracle.load("./models/llama-3.2-1b")
    text = await oracle.generate(system="You are...", user="Attack prompt")
    states = oracle.get_hidden_states()   # List[Tensor(seq, hidden)]
    """

    def __init__(self, tokenizer, model, device: str = "cuda") -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        # Caches for the most recent forward pass — cleared on each generate()
        self._last_hidden_states: list | None = None
        self._last_router_logits: list | None = None
        self._last_past_key_values: Any | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, model_path: str, device: str | None = None) -> "LocalLlamaOracle":
        """Load model from disk and return a ready-to-use oracle.

        Args:
            model_path: Path to a local model directory (downloaded from HF).
            device: 'cuda', 'cpu', or None (auto-detect: cuda if available).
        """
        import torch

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                logger.warning(
                    "Neither CUDA nor MPS available — running on CPU. Inference will be slow. "
                    "Install torch with CUDA: see pyproject.toml [tool.uv.sources]."
                )
        tokenizer, model = _load_model(model_path, device)
        return cls(tokenizer, model, device)

    # ------------------------------------------------------------------
    # BaseModelClient interface
    # ------------------------------------------------------------------

    @property
    def supports_telemetry(self) -> bool:
        return True

    @property
    def model_id(self) -> str:
        cfg = getattr(self._model, "config", None)
        return getattr(cfg, "_name_or_path", "local-llama") or "local-llama"

    async def generate(
        self, system: str, user: str, temperature: float = 0.7, max_new_tokens: int = 512
    ) -> str:
        """Generate text and cache hidden states for telemetry.

        Runs the forward pass synchronously in a thread-pool executor so
        it doesn't block the asyncio event loop.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._generate_sync, system, user, temperature, max_new_tokens
        )
        return result

    async def generate_one_step(
        self,
        system: str | None = None,
        user: str | None = None,
        input_ids: "torch.Tensor | list[int] | None" = None,
        past_key_values: Any | None = None,
        temperature: float = 0.0,
    ) -> dict:
        """Generate exactly one token and return cache-aware step metadata.

        This is the Phase 2 primitive used by Reversible MCTS rollouts.

        Args:
            system: System prompt (required when input_ids is None).
            user: User message (required when input_ids is None).
            input_ids: Optional token ids for step-level continuation. If provided,
                       this call does not re-tokenize system/user.
            past_key_values: Optional cache to continue from. If omitted, uses
                             the most recently cached KV state.
            temperature: Sampling temperature. 0.0 = greedy argmax.

        Returns:
            Dict with keys: next_token_id, next_token_text, input_ids,
            logits, and past_key_values.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_one_step_sync,
            system,
            user,
            input_ids,
            past_key_values,
            temperature,
        )

    # ------------------------------------------------------------------
    # Telemetry accessors
    # ------------------------------------------------------------------

    def get_hidden_states(self) -> list:
        """Return hidden states from the most recent generate() call.

        Returns:
            List of Tensors, one per layer (including embedding layer).
            Shape per tensor: (sequence_length, hidden_size).
            Empty list if generate() has not been called yet.

        Used by HNeuronMonitor (dense path) and LAT probe training.
        """
        if self._last_hidden_states is None:
            return []
        return self._last_hidden_states

    def get_router_logits(self) -> list:
        """Return MoE router logits from the most recent generate() call.

        Returns:
            List of Tensors, one per MoE layer.
            Empty list for dense models (Llama) — always empty in Phase A.

        Non-empty only for gpt-oss-20b (Phase B) when loaded with
        output_router_logits=True. HNeuronMonitor checks len() > 0 to
        select the MoE entropy path vs. dense MLP path.
        """
        if self._last_router_logits is None:
            return []
        return self._last_router_logits

    def get_kv_cache(self) -> Any | None:
        """Return cached past_key_values from the most recent forward pass."""
        return self._last_past_key_values

    def set_kv_cache(self, past_key_values: Any | None) -> None:
        """Override cached past_key_values for the next step-level forward pass."""
        self._last_past_key_values = past_key_values

    def clear_cache(self) -> None:
        """Discard cached hidden states to free memory between episodes."""
        self._last_hidden_states = None
        self._last_router_logits = None
        self._last_past_key_values = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_sync(
        self, system: str, user: str, temperature: float, max_new_tokens: int
    ) -> str:
        """Blocking forward pass — call via run_in_executor from generate()."""
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # Apply chat template if available, otherwise format manually
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except _TEMPLATE_ERRORS as e:
            logger.warning(
                "Chat template failed (%s: %s), falling back to manual format.",
                type(e).__name__, e,
            )
            prompt = f"<|system|>{system}<|user|>{user}<|assistant|>"

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens (exclude the prompt)
        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[0][input_len:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Second forward pass on the full generated sequence to capture hidden states.
        # This is more reliable than output_hidden_states in generate() which
        # is not supported in newer transformers versions.
        try:
            with torch.no_grad():
                hs_outputs = self._model(
                    generated_ids,
                    output_hidden_states=True,
                )
            # hidden_states: tuple of [num_layers+1] Tensors, each [batch, seq, hidden]
            # Take last token representation from each layer → [hidden]
            self._last_hidden_states = [
                h[0, -1, :].detach().cpu()
                for h in hs_outputs.hidden_states
            ]
            # Router logits: only for MoE models (gpt-oss-20b Phase B)
            self._last_router_logits = []
            if hasattr(hs_outputs, "router_logits") and hs_outputs.router_logits:
                self._last_router_logits = [
                    rl.detach().cpu() for rl in hs_outputs.router_logits
                    if rl is not None
                ]
            self._last_past_key_values = hs_outputs.past_key_values
        except Exception as e:
            logger.warning("Hidden state extraction failed: %s", e)
            self._last_hidden_states = []
            self._last_router_logits = []
            self._last_past_key_values = None

        return text

    def _generate_one_step_sync(
        self,
        system: str | None,
        user: str | None,
        input_ids: "torch.Tensor | list[int] | None",
        past_key_values: Any | None,
        temperature: float,
    ) -> dict:
        """Blocking one-token generation step for cache-aware rollouts."""
        import torch

        if input_ids is None:
            if system is None or user is None:
                raise ValueError(
                    "generate_one_step requires (system, user) when input_ids is not provided"
                )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(
                    "Chat template failed (%s: %s), falling back to manual format.",
                    type(e).__name__, e,
                )
                prompt = f"<|system|>{system}<|user|>{user}<|assistant|>"
            model_inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        else:
            if not torch.is_tensor(input_ids):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            model_inputs = {"input_ids": input_ids.to(self._device)}

        if past_key_values is None:
            past_key_values = self._last_past_key_values

        with torch.no_grad():
            outputs = self._model(
                **model_inputs,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            logits = outputs.logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)
            else:
                next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)

        next_token_text = self._tokenizer.decode(
            next_token_ids[0].tolist(),
            skip_special_tokens=True,
        ).strip()

        self._last_hidden_states = [
            h[0, -1, :].detach().cpu()
            for h in outputs.hidden_states
        ]
        self._last_router_logits = []
        if hasattr(outputs, "router_logits") and outputs.router_logits:
            self._last_router_logits = [
                rl.detach().cpu() for rl in outputs.router_logits
                if rl is not None
            ]
        self._last_past_key_values = outputs.past_key_values

        return {
            "next_token_id": int(next_token_ids[0].item()),
            "next_token_text": next_token_text,
            "input_ids": model_inputs["input_ids"].detach().cpu(),
            "logits": logits.detach().cpu(),
            "past_key_values": outputs.past_key_values,
        }
