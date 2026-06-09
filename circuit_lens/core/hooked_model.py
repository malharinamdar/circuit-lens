"""
Circuit-Lens: HookedGPT — A hook-based wrapper around nanoGPT for
Mechanistic Interpretability of Indic Small Language Models.

This module wraps the original nanoGPT model with forward hooks that
capture intermediate activations (residual stream, attention patterns,
MLP outputs, logits at each layer) without modifying the original code.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from collections import OrderedDict
import copy


#  Re-use original nanoGPT components


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = False  # DISABLE flash for interpretability — we need attn weights

        # Always register the causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x, return_attn=False):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Manual attention (no flash) so we can capture attention weights
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        if return_attn:
            return y, att
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class HookedBlock(nn.Module):
    """Transformer block that can return intermediate activations."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, return_components=False):
        ln1_out = self.ln_1(x)
        attn_out, attn_weights = self.attn(ln1_out, return_attn=True)
        x_after_attn = x + attn_out

        ln2_out = self.ln_2(x_after_attn)
        mlp_out = self.mlp(ln2_out)
        x_after_mlp = x_after_attn + mlp_out

        if return_components:
            return x_after_mlp, {
                "ln1_out": ln1_out,
                "attn_out": attn_out,
                "attn_weights": attn_weights,
                "residual_after_attn": x_after_attn,
                "ln2_out": ln2_out,
                "mlp_out": mlp_out,
                "residual_after_mlp": x_after_mlp,
            }
        return x_after_mlp


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class HookedGPT(nn.Module):
    """
    A nanoGPT model with hooks for Mechanistic Interpretability.

    Key features over vanilla nanoGPT:
    - Captures attention patterns at every layer
    - Stores residual stream at every layer boundary
    - Supports "logit lens" (projecting intermediate residuals to vocab)
    - Supports activation patching / steering
    - Supports neuron-level ablation
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([HookedBlock(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        #  Interpretability state
        self._cache: Dict[str, torch.Tensor] = {}
        self._hooks_enabled = True
        self._steering_vectors: Dict[int, torch.Tensor] = {}  # layer_idx vector
        self._ablation_spec: Dict[str, list] = {}  # "layer.neuron" list of neuron indices

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    #  Loading from nanoGPT checkpoint

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: str = "cpu"):
        """Load a nanoGPT-style checkpoint into a HookedGPT model.

        nanoGPT checkpoints contain:
          - 'model': state_dict (may have _orig_mod. prefix from torch.compile)
          - 'model_args': dict with n_layer, n_head, n_embd, block_size, bias, vocab_size, dropout
        """
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        state_dict = checkpoint["model"]

        # Unwrap torch.compile prefix
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        #  Infer config
        # Primary: use model_args saved by nanoGPT's train.py
        model_args = checkpoint.get("model_args", {})

        # Fallback: infer from weights
        vocab_size = model_args.get(
            "vocab_size", state_dict["transformer.wte.weight"].shape[0]
        )
        n_embd = model_args.get(
            "n_embd", state_dict["transformer.wte.weight"].shape[1]
        )
        block_size = model_args.get(
            "block_size", state_dict["transformer.wpe.weight"].shape[0]
        )

        # Count layers from weights as fallback
        if "n_layer" in model_args:
            n_layer = model_args["n_layer"]
        else:
            layer_keys = [k for k in state_dict if k.startswith("transformer.h.")]
            layer_indices = set(int(k.split(".")[2]) for k in layer_keys)
            n_layer = max(layer_indices) + 1

        # n_head: from model_args, else default 8
        n_head = model_args.get("n_head", 8)

        # Check if bias is used
        has_bias = model_args.get(
            "bias", "transformer.h.0.attn.c_attn.bias" in state_dict
        )
        dropout = model_args.get("dropout", 0.0)

        config = GPTConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=has_bias,
        )

        model = cls(config)
        # Load weights with matching
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        print(f"[HookedGPT] Loaded: {model.get_num_params()/1e6:.2f}M params")
        print(f"  n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")
        print(f"  vocab_size={vocab_size}, block_size={block_size}")

        return model

    #  Core forward with caching

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cache_activations: bool = False,
    ):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        if cache_activations and self._hooks_enabled:
            self._cache = {
                "embeddings": (tok_emb + pos_emb).detach(),
                "residual_stream": [],
                "attention_patterns": [],
                "attn_outputs": [],
                "mlp_outputs": [],
                "logit_lens": [],
            }

        for i, block in enumerate(self.transformer.h):
            # Apply steering vector if set
            if i in self._steering_vectors:
                x = x + self._steering_vectors[i].to(device)

            if cache_activations and self._hooks_enabled:
                x, components = block(x, return_components=True)
                self._cache["residual_stream"].append(x.detach())
                self._cache["attention_patterns"].append(
                    components["attn_weights"].detach()
                )
                self._cache["attn_outputs"].append(components["attn_out"].detach())
                self._cache["mlp_outputs"].append(components["mlp_out"].detach())

                # Logit lens: project intermediate residual to vocab
                with torch.no_grad():
                    normed = self.transformer.ln_f(x)
                    logits_at_layer = self.lm_head(normed)
                    self._cache["logit_lens"].append(logits_at_layer.detach())
            else:
                x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        elif cache_activations:
            # When caching, return full-sequence logits so logit_lens
            # and activation patching can index any position.
            logits = self.lm_head(x)
            loss = None
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if cache_activations and self._hooks_enabled:
            self._cache["final_logits"] = logits.detach()

        return logits, loss

    #  Interpretability API

    def get_cache(self) -> Dict[str, any]:
        """Return the activation cache from the last forward pass."""
        return self._cache

    def run_with_cache(
        self, idx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """Run forward pass and return (logits, cache)."""
        logits, _ = self.forward(idx, cache_activations=True)
        return logits, self._cache

    def logit_lens(
        self, idx: torch.Tensor, tokenizer, top_k: int = 5, position: int = -1
    ) -> List[Dict]:
        """
        Logit Lens: see what the model "thinks" the next token is at each layer.

        Returns list of dicts (one per layer) with top-k token predictions.
        """
        self.run_with_cache(idx)
        results = []

        for layer_idx, layer_logits in enumerate(self._cache["logit_lens"]):
            # layer_logits: (B, T, vocab_size)
            logits_at_pos = layer_logits[0, position, :]  # first batch, target position
            probs = F.softmax(logits_at_pos, dim=-1)
            topk_probs, topk_ids = torch.topk(probs, top_k)

            layer_result = {
                "layer": layer_idx,
                "top_tokens": [],
            }
            for prob, tok_id in zip(topk_probs, topk_ids):
                token_str = tokenizer.decode([tok_id.item()])
                layer_result["top_tokens"].append(
                    {"token": token_str, "id": tok_id.item(), "prob": prob.item()}
                )
            results.append(layer_result)

        return results

    def get_attention_patterns(
        self, idx: torch.Tensor
    ) -> List[torch.Tensor]:
        """Get attention patterns for all layers. Returns list of (B, n_head, T, T)."""
        self.run_with_cache(idx)
        return self._cache["attention_patterns"]

    #  Activation Steering

    def set_steering_vector(self, layer_idx: int, vector: torch.Tensor):
        """Add a steering vector to a specific layer's residual stream."""
        self._steering_vectors[layer_idx] = vector

    def clear_steering_vectors(self):
        self._steering_vectors = {}

    #  Activation Patching

    def patch_activations(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        patch_layer: int,
        patch_position: int,
        patch_type: str = "residual",  # "residual", "attn", "mlp"
    ) -> torch.Tensor:
        """
        Activation patching: run corrupted input but replace one
        component at (layer, position) with the clean activation.

        Returns patched logits.
        """
        # Run clean to get activations
        _, clean_cache = self.run_with_cache(clean_input)

        # Run corrupted with manual patching
        device = corrupted_input.device
        b, t = corrupted_input.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(corrupted_input)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for i, block in enumerate(self.transformer.h):
            x, components = block(x, return_components=True)

            if i == patch_layer:
                if patch_type == "residual":
                    clean_residual = clean_cache["residual_stream"][i]
                    x[:, patch_position, :] = clean_residual[:, patch_position, :]
                elif patch_type == "attn":
                    clean_attn = clean_cache["attn_outputs"][i]
                    # Re-compute: x = x_before_attn + attn_out + mlp_out
                    # We patch the attn contribution
                    diff = (
                        clean_attn[:, patch_position, :]
                        - components["attn_out"][:, patch_position, :]
                    )
                    x[:, patch_position, :] += diff
                elif patch_type == "mlp":
                    clean_mlp = clean_cache["mlp_outputs"][i]
                    diff = (
                        clean_mlp[:, patch_position, :]
                        - components["mlp_out"][:, patch_position, :]
                    )
                    x[:, patch_position, :] += diff

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x[:, [-1], :])
        return logits

    #  Neuron Analysis

    def get_neuron_activations(
        self, idx: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Get MLP neuron activations (pre-GELU) at a specific layer."""
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for i, block in enumerate(self.transformer.h):
            ln1_out = block.ln_1(x)
            attn_out = block.attn(ln1_out)
            x_after_attn = x + attn_out
            ln2_out = block.ln_2(x_after_attn)

            if i == layer_idx:
                # Get pre-GELU activations
                pre_gelu = block.mlp.c_fc(ln2_out)
                return pre_gelu.detach()

            mlp_out = block.mlp(ln2_out)
            x = x_after_attn + mlp_out

        return None

    #  Induction Head Detection

    def detect_induction_heads(
        self, idx: torch.Tensor, threshold: float = 0.4
    ) -> List[Tuple[int, int, float]]:
        """
        Detect previous-token heads by checking if attention concentrates
        on position i-1 for each position i.

        For true induction head detection on repeated sequences, use
        InductionHeadAnalyzer which checks the [A][B]...[A] attend-to-B
        pattern on the second repeat.

        Returns list of (layer, head, score) tuples.
        """
        attn_patterns = self.get_attention_patterns(idx)
        candidates = []

        for layer_idx, attn in enumerate(attn_patterns):
            # attn: (B, n_head, T, T)
            T = attn.shape[-1]
            if T < 4:
                continue

            for head_idx in range(attn.shape[1]):
                head_attn = attn[0, head_idx]  # (T, T)

                # Induction head score: how much does position i attend to
                # positions that follow tokens identical to the token at i-1?
                # Simplified: check diagonal offset pattern
                score = 0.0
                count = 0
                for pos in range(2, T):
                    # Check if attention at pos is concentrated on pos-1
                    # (previous-token head behavior)
                    prev_attn = head_attn[pos, pos - 1].item()
                    score += prev_attn
                    count += 1

                avg_score = score / max(count, 1)
                if avg_score > threshold:
                    candidates.append((layer_idx, head_idx, avg_score))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    #  Generation with interpretability

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        cache_last_step: bool = False,
    ) -> torch.Tensor:
        """Generate tokens, optionally caching the last step's activations."""
        for step in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]

            is_last = step == max_new_tokens - 1
            logits, _ = self.forward(
                idx_cond, cache_activations=(cache_last_step and is_last)
            )
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)

        return idx
