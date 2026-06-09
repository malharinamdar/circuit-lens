"""
Circuit-Lens: Analysis Module
Implements key mechanistic interpretability experiments for Indic SLMs.

Experiments:
1. Induction Head Analysis
2. Gender-Verb Agreement Circuit Detection (Hindi/Marathi specific)
3. Logit Lens Analysis
4. Attention Pattern Visualization Data
5. Neuron-level Analysis for Morphological Features
6. Activation Patching for Circuit Identification
7. Cross-lingual Circuit Comparison
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# 1. INDUCTION HEAD ANALYSIS

class InductionHeadAnalyzer:
    """
    Detects and analyzes induction heads in Indic SLMs.

    Induction heads implement the pattern:
    [A][B]...[A] predict [B]
    They are fundamental to in-context learning.

    Key question: Do induction heads form earlier/later in Indic models
    compared to English models of similar size?
    """

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def create_repeated_sequence(self, text: str, repeat: int = 2) -> torch.Tensor:
        """Create a repeated token sequence for induction head detection."""
        tokens = self.tokenizer.encode(text)
        repeated = tokens * repeat
        return torch.tensor([repeated], dtype=torch.long, device=self.device)

    def compute_induction_scores(
        self, input_ids: torch.Tensor
    ) -> np.ndarray:
        """
        Compute induction head scores for all (layer, head) pairs.

        Score = average attention from position i to position (i - seq_len + 1)
        on the repeated half of the sequence.

        Returns: (n_layer, n_head) array of scores.
        """
        attn_patterns = self.model.get_attention_patterns(input_ids)
        n_layer = len(attn_patterns)
        n_head = attn_patterns[0].shape[1]
        T = attn_patterns[0].shape[-1]
        half_T = T // 2

        scores = np.zeros((n_layer, n_head))

        for layer_idx, attn in enumerate(attn_patterns):
            for head_idx in range(n_head):
                head_attn = attn[0, head_idx].cpu().numpy()  # (T, T)

                # Induction head pattern on repeated sequence [A B C][A B C]:
                # At position i in the second repeat (i >= half_T),
                # the induction head should attend to position (i - half_T + 1)
                # i.e., the token that FOLLOWED the matching token in first half.
                #
                # Example: tokens = [A B C A B C], half_T = 3
                #   pos 3 (second A): should attend to pos 1 (B, the token after first A)
                #   pos 4 (second B): should attend to pos 2 (C, the token after first B)
                score_sum = 0.0
                count = 0
                for pos in range(half_T, T):
                    target_pos = pos - half_T + 1
                    if 0 <= target_pos < T:
                        score_sum += head_attn[pos, target_pos]
                        count += 1

                scores[layer_idx, head_idx] = score_sum / max(count, 1)

        return scores

    def run_analysis(
        self,
        test_sentences: List[str],
        threshold: float = 0.3,
    ) -> Dict:
        """
        Run full induction head analysis on multiple test sentences.

        Returns a dict with per-sentence scores and aggregated results.
        """
        all_scores = []
        for sent in test_sentences:
            input_ids = self.create_repeated_sequence(sent)
            scores = self.compute_induction_scores(input_ids)
            all_scores.append(scores)

        avg_scores = np.mean(all_scores, axis=0)

        # Find induction heads above threshold
        induction_heads = []
        for layer in range(avg_scores.shape[0]):
            for head in range(avg_scores.shape[1]):
                if avg_scores[layer, head] > threshold:
                    induction_heads.append({
                        "layer": int(layer),
                        "head": int(head),
                        "score": float(avg_scores[layer, head]),
                    })

        induction_heads.sort(key=lambda x: x["score"], reverse=True)

        return {
            "score_matrix": avg_scores.tolist(),
            "induction_heads": induction_heads,
            "n_layer": int(avg_scores.shape[0]),
            "n_head": int(avg_scores.shape[1]),
            "threshold": threshold,
        }


# 2. GENDER-VERB AGREEMENT CIRCUIT DETECTION

class GenderAgreementAnalyzer:
    """
    Identifies circuits responsible for gender-verb agreement in Hindi/Marathi.

    Hindi examples:
    - "लड़का जाता है" (ladka jaata hai — boy goes, masculine)
    - "लड़की जाती है" (ladki jaati hai — girl goes, feminine)

    Marathi examples:
    - "मुलगा जातो" (mulga jaato — boy goes, masculine)
    - "मुलगी जाते" (mulgi jaate — girl goes, feminine)

    Method: Activation patching — swap the gender-carrying token's
    activations between masc/fem sentences and observe which layers
    cause the verb form to flip.
    """

    # Pre-defined gender agreement test pairs
    HINDI_PAIRS = [
        {
            "masc": "लड़का जाता है",
            "fem": "लड़की जाती है",
            "verb_masc": "जाता",
            "verb_fem": "जाती",
            "description": "go (jana)",
        },
        {
            "masc": "लड़का खाता है",
            "fem": "लड़की खाती है",
            "verb_masc": "खाता",
            "verb_fem": "खाती",
            "description": "eat (khana)",
        },
        {
            "masc": "लड़का पढ़ता है",
            "fem": "लड़की पढ़ती है",
            "verb_masc": "पढ़ता",
            "verb_fem": "पढ़ती",
            "description": "read/study (padhna)",
        },
        {
            "masc": "राम सोता है",
            "fem": "सीता सोती है",
            "verb_masc": "सोता",
            "verb_fem": "सोती",
            "description": "sleep (sona)",
        },
        {
            "masc": "मोहन गाता है",
            "fem": "गीता गाती है",
            "verb_masc": "गाता",
            "verb_fem": "गाती",
            "description": "sing (gaana)",
        },
    ]

    MARATHI_PAIRS = [
        {
            "masc": "मुलगा जातो",
            "fem": "मुलगी जाते",
            "verb_masc": "जातो",
            "verb_fem": "जाते",
            "description": "go (jana)",
        },
        {
            "masc": "मुलगा खातो",
            "fem": "मुलगी खाते",
            "verb_masc": "खातो",
            "verb_fem": "खाते",
            "description": "eat (khane)",
        },
    ]

    def __init__(self, model, tokenizer, language="hindi", device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.language = language
        self.device = device
        self.pairs = self.HINDI_PAIRS if language == "hindi" else self.MARATHI_PAIRS

    def _encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)

    def _get_verb_token_id(self, verb: str) -> int:
        """Get the first token ID of a verb form, skipping special tokens."""
        ids = self.tokenizer.encode(verb, add_special_tokens=False)
        return ids[0] if ids else -1

    def analyze_pair(self, pair: Dict) -> Dict:
        """
        For a single gender pair, perform activation patching across all layers.

        Returns a dict mapping layer change in logit difference when patching.
        """
        masc_ids = self._encode(pair["masc"])
        fem_ids = self._encode(pair["fem"])

        # Get target verb token IDs
        verb_masc_id = self._get_verb_token_id(pair["verb_masc"])
        verb_fem_id = self._get_verb_token_id(pair["verb_fem"])

        if verb_masc_id == -1 or verb_fem_id == -1:
            return {"error": "Could not tokenize verb forms"}

        # Baseline: logit difference on clean inputs
        with torch.no_grad():
            masc_logits, _ = self.model(masc_ids, cache_activations=True)
            masc_logit_diff = (
                masc_logits[0, -1, verb_masc_id] - masc_logits[0, -1, verb_fem_id]
            ).item()

            fem_logits, _ = self.model(fem_ids, cache_activations=True)
            fem_logit_diff = (
                fem_logits[0, -1, verb_fem_id] - fem_logits[0, -1, verb_masc_id]
            ).item()

        # Activation patching: for each layer, patch the first token's
        # activations from feminine into masculine and check verb prediction
        n_layers = self.model.config.n_layer
        patching_results = {}

        for layer_idx in range(n_layers):
            for patch_type in ["residual", "attn", "mlp"]:
                with torch.no_grad():
                    patched_logits = self.model.patch_activations(
                        clean_input=fem_ids,
                        corrupted_input=masc_ids,
                        patch_layer=layer_idx,
                        patch_position=0,  # Subject noun position
                        patch_type=patch_type,
                    )
                    patched_diff = (
                        patched_logits[0, -1, verb_masc_id]
                        - patched_logits[0, -1, verb_fem_id]
                    ).item()

                key = f"L{layer_idx}_{patch_type}"
                patching_results[key] = {
                    "layer": layer_idx,
                    "patch_type": patch_type,
                    "baseline_logit_diff": masc_logit_diff,
                    "patched_logit_diff": patched_diff,
                    "change": patched_diff - masc_logit_diff,
                    "normalized_change": (patched_diff - masc_logit_diff) / max(abs(masc_logit_diff), 1e-6),
                }

        return {
            "pair": pair,
            "masc_logit_diff": masc_logit_diff,
            "fem_logit_diff": fem_logit_diff,
            "patching_results": patching_results,
        }

    def run_full_analysis(self) -> Dict:
        """Run gender agreement analysis on all pairs."""
        results = []
        for pair in self.pairs:
            result = self.analyze_pair(pair)
            results.append(result)

        # Aggregate: which layers matter most for gender agreement?
        layer_importance = defaultdict(lambda: {"residual": 0, "attn": 0, "mlp": 0, "count": 0})
        for r in results:
            if "error" in r:
                continue
            for key, val in r["patching_results"].items():
                layer = val["layer"]
                ptype = val["patch_type"]
                layer_importance[layer][ptype] += abs(val["normalized_change"])
                layer_importance[layer]["count"] += 1

        # Normalize
        for layer in layer_importance:
            count = max(layer_importance[layer]["count"], 1)
            for ptype in ["residual", "attn", "mlp"]:
                layer_importance[layer][ptype] /= (count / 3)  # 3 patch types

        return {
            "language": self.language,
            "per_pair_results": results,
            "layer_importance": dict(layer_importance),
        }


# 3. LOGIT LENS ANALYSIS

class LogitLensAnalyzer:
    """
    Logit Lens: At each layer, project the residual stream to vocab space
    and see what the model "thinks" the next token is.

    This reveals how predictions evolve through the network depth.
    """

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def analyze(
        self, text: str, top_k: int = 10, target_position: int = -1
    ) -> Dict:
        """Run logit lens on a text input."""
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
        tokens = [self.tokenizer.decode([t]) for t in input_ids[0]]

        results = self.model.logit_lens(input_ids, self.tokenizer, top_k=top_k, position=target_position)

        return {
            "input_text": text,
            "input_tokens": tokens,
            "target_position": target_position,
            "layers": results,
        }

    def analyze_all_positions(self, text: str, top_k: int = 5) -> Dict:
        """Run logit lens for every position in the input."""
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
        tokens = [self.tokenizer.decode([t]) for t in input_ids[0]]
        seq_len = input_ids.shape[1]

        # Run once with cache
        self.model.run_with_cache(input_ids)
        cache = self.model.get_cache()

        all_positions = []
        for pos in range(seq_len):
            layers_at_pos = []
            for layer_idx, layer_logits in enumerate(cache["logit_lens"]):
                logits_at_pos = layer_logits[0, pos, :]
                probs = F.softmax(logits_at_pos, dim=-1)
                topk_probs, topk_ids = torch.topk(probs, top_k)

                layer_data = {
                    "layer": layer_idx,
                    "top_tokens": [
                        {
                            "token": self.tokenizer.decode([tid.item()]),
                            "prob": p.item(),
                        }
                        for p, tid in zip(topk_probs, topk_ids)
                    ],
                }
                layers_at_pos.append(layer_data)

            all_positions.append({
                "position": pos,
                "input_token": tokens[pos],
                "layers": layers_at_pos,
            })

        return {
            "input_text": text,
            "input_tokens": tokens,
            "positions": all_positions,
        }


# 4. ATTENTION PATTERN ANALYSIS

class AttentionAnalyzer:
    """Analyze attention patterns across layers and heads."""

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def get_attention_data(self, text: str) -> Dict:
        """Get full attention data for visualization."""
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
        tokens = [self.tokenizer.decode([t]) for t in input_ids[0]]

        attn_patterns = self.model.get_attention_patterns(input_ids)

        layers_data = []
        for layer_idx, attn in enumerate(attn_patterns):
            heads_data = []
            for head_idx in range(attn.shape[1]):
                head_attn = attn[0, head_idx].cpu().numpy().tolist()
                heads_data.append({
                    "head": head_idx,
                    "attention_matrix": head_attn,
                })
            layers_data.append({
                "layer": layer_idx,
                "heads": heads_data,
            })

        return {
            "input_text": text,
            "tokens": tokens,
            "layers": layers_data,
        }

    def classify_head_types(self, text: str) -> Dict:
        """
        Classify each attention head into categories:
        - Previous token head
        - Induction head
        - Position head (attends to specific positions)
        - Content head (attends based on content)
        """
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
        attn_patterns = self.model.get_attention_patterns(input_ids)

        head_types = {}
        for layer_idx, attn in enumerate(attn_patterns):
            T = attn.shape[-1]
            for head_idx in range(attn.shape[1]):
                head_attn = attn[0, head_idx].cpu().numpy()

                # Previous token score: avg attention to pos-1
                prev_token_score = 0
                for i in range(1, T):
                    prev_token_score += head_attn[i, i - 1]
                prev_token_score /= max(T - 1, 1)

                # BOS/first token attention score
                bos_score = np.mean(head_attn[:, 0])

                # Diagonal (self-attention) score
                diag_score = np.mean(np.diag(head_attn))

                # Entropy of attention (higher = more diffuse)
                entropy = 0
                for i in range(T):
                    row = head_attn[i, :i+1]
                    row = row + 1e-10
                    entropy -= np.sum(row * np.log(row))
                entropy /= max(T, 1)

                head_type = "other"
                if prev_token_score > 0.5:
                    head_type = "previous_token"
                elif bos_score > 0.5:
                    head_type = "bos_attending"
                elif diag_score > 0.5:
                    head_type = "self_attending"
                elif entropy > 3.0:
                    head_type = "diffuse"

                head_types[f"L{layer_idx}H{head_idx}"] = {
                    "layer": layer_idx,
                    "head": head_idx,
                    "type": head_type,
                    "prev_token_score": float(prev_token_score),
                    "bos_score": float(bos_score),
                    "diag_score": float(diag_score),
                    "entropy": float(entropy),
                }

        return head_types


# 5. NEURON ANALYSIS

class NeuronAnalyzer:
    """
    Analyze individual MLP neurons for feature detection.
    Find neurons that activate for specific linguistic features.
    """

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def find_feature_neurons(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
        layer_idx: int,
    ) -> List[Dict]:
        """
        Find neurons that activate more for positive examples than negative.
        Useful for finding gender-specific neurons, formality neurons, etc.
        """
        pos_activations = []
        neg_activations = []

        for text in positive_examples:
            input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
            with torch.no_grad():
                acts = self.model.get_neuron_activations(input_ids, layer_idx)
                # Mean over positions
                pos_activations.append(acts[0].mean(dim=0).cpu())

        for text in negative_examples:
            input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
            with torch.no_grad():
                acts = self.model.get_neuron_activations(input_ids, layer_idx)
                neg_activations.append(acts[0].mean(dim=0).cpu())

        pos_mean = torch.stack(pos_activations).mean(dim=0)
        neg_mean = torch.stack(neg_activations).mean(dim=0)

        # Difference in activation
        diff = pos_mean - neg_mean

        # Top discriminative neurons
        top_k = 20
        top_vals, top_indices = torch.topk(diff.abs(), top_k)

        neurons = []
        for val, idx in zip(top_vals, top_indices):
            neurons.append({
                "neuron_idx": int(idx.item()),
                "activation_diff": float(diff[idx].item()),
                "abs_diff": float(val.item()),
                "pos_mean_activation": float(pos_mean[idx].item()),
                "neg_mean_activation": float(neg_mean[idx].item()),
            })

        return neurons

    def find_gender_neurons(self, layer_idx: int) -> List[Dict]:
        """Find neurons specific to masculine vs feminine forms in Hindi."""
        masc_examples = [
            "लड़का जाता है",
            "राम खाता है",
            "मोहन पढ़ता है",
            "वह सोता है",
            "बच्चा खेलता है",
        ]
        fem_examples = [
            "लड़की जाती है",
            "सीता खाती है",
            "गीता पढ़ती है",
            "वह सोती है",
            "बच्ची खेलती है",
        ]
        return self.find_feature_neurons(masc_examples, fem_examples, layer_idx)


# 6. CROSS-LINGUAL COMPARISON

class CrossLingualAnalyzer:
    """
    Compare circuit structures across Hindi, Marathi, Bangla models.
    Key questions:
    - Do induction heads appear at similar layers?
    - Are gender agreement circuits similar across languages?
    - How do attention patterns differ for similar prompts?
    """

    def __init__(self, models: Dict, tokenizers, device="cpu"):
        """
        models: {"hindi": model, "marathi": model, ...}
        tokenizers: either a single tokenizer (shared across languages)
                    or a dict {"hindi": tokenizer, "marathi": tokenizer, ...}
        """
        self.models = models
        # Normalize: if a single tokenizer is passed, replicate for each lang
        if isinstance(tokenizers, dict):
            self.tokenizers = tokenizers
        else:
            self.tokenizers = {lang: tokenizers for lang in models}
        self.device = device

    def compare_induction_heads(self, test_texts: Dict[str, List[str]]) -> Dict:
        """
        Compare induction head formation across languages.
        test_texts: {"hindi": [...], "marathi": [...], ...}
        """
        results = {}
        for lang, texts in test_texts.items():
            if lang not in self.models:
                continue
            analyzer = InductionHeadAnalyzer(
                self.models[lang], self.tokenizers[lang], self.device
            )
            results[lang] = analyzer.run_analysis(texts)

        return results

    def compare_attention_entropy(self, texts: Dict[str, str]) -> Dict:
        """Compare attention entropy across languages for equivalent prompts."""
        results = {}
        for lang, text in texts.items():
            if lang not in self.models:
                continue
            model = self.models[lang]
            tokenizer = self.tokenizers[lang]
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
            attn_patterns = model.get_attention_patterns(input_ids)

            layer_entropies = []
            for layer_idx, attn in enumerate(attn_patterns):
                T = attn.shape[-1]
                head_entropies = []
                for h in range(attn.shape[1]):
                    head_attn = attn[0, h].cpu().numpy()
                    entropy = 0
                    for i in range(T):
                        row = head_attn[i, :i+1] + 1e-10
                        entropy -= np.sum(row * np.log(row))
                    head_entropies.append(entropy / max(T, 1))
                layer_entropies.append({
                    "layer": layer_idx,
                    "mean_entropy": float(np.mean(head_entropies)),
                    "head_entropies": [float(e) for e in head_entropies],
                })

            results[lang] = {
                "text": text,
                "layer_entropies": layer_entropies,
            }

        return results


# 7. ACTIVATION STEERING

class ActivationSteeringAnalyzer:
    """
    Compute and apply activation steering vectors.
    E.g., formal ↔ informal, masculine ↔ feminine.
    """

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def compute_steering_vector(
        self,
        positive_texts: List[str],
        negative_texts: List[str],
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Compute a steering vector as the mean difference in residual stream
        activations between positive and negative examples.
        """
        pos_residuals = []
        neg_residuals = []

        for text in positive_texts:
            input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
            with torch.no_grad():
                _, cache = self.model.run_with_cache(input_ids)
                # Mean over positions
                residual = cache["residual_stream"][layer_idx][0].mean(dim=0)
                pos_residuals.append(residual)

        for text in negative_texts:
            input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
            with torch.no_grad():
                _, cache = self.model.run_with_cache(input_ids)
                residual = cache["residual_stream"][layer_idx][0].mean(dim=0)
                neg_residuals.append(residual)

        pos_mean = torch.stack(pos_residuals).mean(dim=0)
        neg_mean = torch.stack(neg_residuals).mean(dim=0)

        steering_vector = pos_mean - neg_mean
        return steering_vector

    def steer_generation(
        self,
        prompt: str,
        steering_vector: torch.Tensor,
        layer_idx: int,
        alpha: float = 1.0,
        max_new_tokens: int = 100,
    ) -> str:
        """Generate text with a steering vector applied."""
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)

        # Apply steering
        scaled_vector = (alpha * steering_vector).unsqueeze(0).unsqueeze(0)
        self.model.set_steering_vector(layer_idx, scaled_vector)

        output_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Clear steering
        self.model.clear_steering_vectors()

        return output_text
