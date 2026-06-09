#!/usr/bin/env python3
"""
Circuit-Lens: Main Experiment Runner
=====================================
Runs all mechanistic interpretability experiments on Regional-TinyStories
Indic SLMs and exports structured JSON results + visualization data.

Usage:
  python run_experiments.py --ckpt path/to/checkpoint.pt --tokenizer sarvamai/sarvam-1 --lang hindi
  python run_experiments.py --run-all --ckpt-dir path/to/checkpoints/
"""

import argparse
import json
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add parent to path so imports work when run standalone
sys.path.insert(0, str(Path(__file__).resolve().parent))

from circuit_lens.core.hooked_model import HookedGPT
from circuit_lens.analysis.experiments import (
    InductionHeadAnalyzer,
    GenderAgreementAnalyzer,
    LogitLensAnalyzer,
    AttentionAnalyzer,
    NeuronAnalyzer,
    CrossLingualAnalyzer,
    ActivationSteeringAnalyzer,
)


#  Utility

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(ckpt_path: str, tokenizer_name: str, device: str):
    """Load a HookedGPT model from checkpoint and its tokenizer."""
    from transformers import AutoTokenizer

    print(f"  Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"  Loading checkpoint: {ckpt_path}")
    model = HookedGPT.from_checkpoint(ckpt_path, device=device)
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model loaded: {params:.2f}M params on {device}")
    return model, tokenizer


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super().default(obj)


#  Test Prompts per Language

PROMPTS = {
    "hindi": [
        "एक समय की बात है, एक छोटे से गाँव में",
        "एक दिन, एक छोटा बच्चा जंगल में गया",
        "सूरज चमक रहा था और बच्चे खेल रहे थे",
        "राम और सीता बहुत अच्छे दोस्त थे",
        "एक सुंदर तितली फूलों पर बैठी थी",
    ],
    "marathi": [
        "एकदा एका छोट्या गावात एक मुलगा राहत होता",
        "सूर्य उगवत होता आणि पक्षी गात होते",
        "एक दिवस, एक छोटा मुलगा जंगलात गेला",
        "आई आणि बाबा खूप आनंदी होते",
        "एक सुंदर फुलपाखरू बागेत उडत होते",
    ],
    "bangla": [
        "একদিন একটি ছোট্ট ছেলে বনে গেল",
        "সূর্য উঠছিল আর পাখিরা গান গাইছিল",
        "রাহুল আর সুমি খুব ভালো বন্ধু ছিল",
        "একটি সুন্দর প্রজাপতি ফুলের উপর বসেছিল",
        "একসময় এক ছোট্ট গ্রামে এক মেয়ে থাকত",
    ],
}

GENDER_PROMPTS_HINDI = {
    "masculine": [
        "लड़का जाता है",
        "लड़का खाता है",
        "लड़का पढ़ता है",
        "लड़का सोता है",
        "लड़का गाता है",
    ],
    "feminine": [
        "लड़की जाती है",
        "लड़की खाती है",
        "लड़की पढ़ती है",
        "लड़की सोती है",
        "लड़की गाती है",
    ],
}

GENDER_PROMPTS_MARATHI = {
    "masculine": [
        "मुलगा जातो",
        "मुलगा खातो",
    ],
    "feminine": [
        "मुलगी जाते",
        "मुलगी खाते",
    ],
}

FORMALITY_PROMPTS = {
    "hindi": {
        "formal": [
            "आदरणीय महोदय, मैं आपको सूचित करना चाहता हूँ",
            "कृपया ध्यान दें कि यह बहुत महत्वपूर्ण है",
            "मैं आपसे निवेदन करता हूँ कि",
        ],
        "informal": [
            "अरे यार, सुन ना एक बात बताता हूँ",
            "चल ना बाहर चलते हैं खेलने",
            "देख ये कितना मज़ेदार है",
        ],
    },
}


#  Individual Experiment Runners

def run_induction_head_analysis(model, tokenizer, lang, device):
    """Run induction head detection experiment."""
    print("\n  Experiment 1: Induction Head Analysis")
    analyzer = InductionHeadAnalyzer(model, tokenizer, device)

    prompts = PROMPTS.get(lang, PROMPTS["hindi"])
    results = analyzer.run_analysis(prompts)

    # Summary
    n_heads = len(results.get("induction_heads", []))
    print(f"    Found {n_heads} induction heads (threshold=0.3)")
    for h in results.get("induction_heads", [])[:5]:
        print(f"      Layer {h['layer']}, Head {h['head']}: score={h['score']:.4f}")

    return results


def run_gender_agreement_analysis(model, tokenizer, lang, device):
    """Run gender-verb agreement circuit detection."""
    if lang not in ("hindi", "marathi"):
        print(f"\n  Experiment 2: Gender Agreement — skipped (lang={lang})")
        return {"status": "skipped", "reason": f"not applicable for {lang}"}

    print(f"\n  Experiment 2: Gender Agreement Circuit ({lang})")
    analyzer = GenderAgreementAnalyzer(model, tokenizer, language=lang, device=device)

    results = analyzer.run_full_analysis()

    # Print layer importance
    layer_imp = results.get("layer_importance", {})
    for layer_idx in sorted(layer_imp.keys()):
        scores = layer_imp[layer_idx]
        residual_change = scores.get("residual", 0)
        if abs(residual_change) > 0.05:
            print(f"    Layer {layer_idx}: residual={residual_change:.4f}, attn={scores.get('attn', 0):.4f}, mlp={scores.get('mlp', 0):.4f}")

    return results


def run_logit_lens_analysis(model, tokenizer, lang, device):
    """Run logit lens to see prediction evolution through layers."""
    print("\n  Experiment 3: Logit Lens")
    analyzer = LogitLensAnalyzer(model, tokenizer, device)

    prompts = PROMPTS.get(lang, PROMPTS["hindi"])[:3]
    all_results = {}

    for prompt in prompts:
        result = analyzer.analyze(prompt, top_k=5)
        all_results[prompt] = result
        print(f"    Prompt: '{prompt[:40]}...'")
        if isinstance(result, dict) and "layers" in result:
            for layer_data in result["layers"][-2:]:  # last 2 layers
                layer_idx = layer_data.get("layer", "?")
                top_tokens = layer_data.get("top_tokens", [])
                if top_tokens:
                    tok = top_tokens[0]
                    print(f"      Layer {layer_idx}: top prediction = '{tok['token']}' ({tok['prob']:.4f})")

    return all_results


def run_attention_analysis(model, tokenizer, lang, device):
    """Classify attention head types across the model."""
    print("\n  Experiment 4: Attention Pattern Classification")
    analyzer = AttentionAnalyzer(model, tokenizer, device)

    prompt = PROMPTS.get(lang, PROMPTS["hindi"])[0]
    results = analyzer.classify_head_types(prompt)

    # Count types
    type_counts = defaultdict(int)
    for head_key, info in results.items():
        type_counts[info["type"]] += 1

    for head_type, count in sorted(type_counts.items()):
        print(f"    {head_type}: {count} heads")

    return {"classifications": results, "type_counts": dict(type_counts), "prompt": prompt}


def run_neuron_analysis(model, tokenizer, lang, device):
    """Find gender-discriminative neurons."""
    if lang not in ("hindi", "marathi"):
        print(f"\n  Experiment 5: Neuron Analysis — skipped (lang={lang})")
        return {"status": "skipped"}

    print(f"\n  Experiment 5: Feature Neuron Analysis ({lang})")
    analyzer = NeuronAnalyzer(model, tokenizer, device)

    results = {}
    n_layers = model.config.n_layer

    # Analyze a few key layers
    for layer_idx in [0, n_layers // 2, n_layers - 1]:
        neurons = analyzer.find_gender_neurons(layer_idx)
        results[f"layer_{layer_idx}"] = neurons
        if neurons:
            top = neurons[:3]
            print(f"    Layer {layer_idx}: top gender neurons = {[(n['neuron_idx'], round(n['activation_diff'], 4)) for n in top]}")

    return results


def run_activation_steering(model, tokenizer, lang, device):
    """Compute formality steering vectors and demo steered generation."""
    if lang not in FORMALITY_PROMPTS:
        print(f"\n  Experiment 7: Activation Steering — skipped (no prompts for {lang})")
        return {"status": "skipped"}

    print(f"\n  Experiment 7: Activation Steering ({lang})")
    analyzer = ActivationSteeringAnalyzer(model, tokenizer, device)

    formal_texts = FORMALITY_PROMPTS[lang]["formal"]
    informal_texts = FORMALITY_PROMPTS[lang]["informal"]

    n_layers = model.config.n_layer
    mid_layer = n_layers // 2

    # Compute steering vector at mid layer
    vector = analyzer.compute_steering_vector(formal_texts, informal_texts, mid_layer)

    results = {"steering_layer": mid_layer, "vector_norm": float(vector.norm().item())}

    # Demo steered generation
    test_prompt = PROMPTS.get(lang, PROMPTS["hindi"])[0]

    for alpha in [0.0, 1.0, 2.0, -1.0]:
        try:
            output = analyzer.steer_generation(
                test_prompt, vector, mid_layer, alpha=alpha, max_new_tokens=60
            )
            results[f"alpha_{alpha}"] = output
            direction = "neutral" if alpha == 0 else ("formal" if alpha > 0 else "informal")
            print(f"    α={alpha:+.1f} ({direction}): '{output[:80]}...'")
        except Exception as e:
            results[f"alpha_{alpha}"] = f"error: {str(e)}"

    return results


#  Main Pipeline

def run_all_experiments(ckpt_path, tokenizer_name, lang, output_dir, device):
    """Run the full experiment suite on a single checkpoint."""
    print(f"\n{'='*70}")
    print(f"  Circuit-Lens: Running experiments")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Tokenizer  : {tokenizer_name}")
    print(f"  Language   : {lang}")
    print(f"  Device     : {device}")
    print(f"{'='*70}")

    model, tokenizer = load_model_and_tokenizer(ckpt_path, tokenizer_name, device)
    os.makedirs(output_dir, exist_ok=True)

    all_results = {
        "metadata": {
            "checkpoint": ckpt_path,
            "tokenizer": tokenizer_name,
            "language": lang,
            "device": device,
            "n_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "n_embd": model.config.n_embd,
            "timestamp": datetime.now().isoformat(),
        }
    }

    t0 = time.time()

    # 1. Induction Heads
    try:
        all_results["induction_heads"] = run_induction_head_analysis(
            model, tokenizer, lang, device
        )
    except Exception as e:
        print(f"    Induction head analysis failed: {e}")
        all_results["induction_heads"] = {"error": str(e)}

    # 2. Gender Agreement
    try:
        all_results["gender_agreement"] = run_gender_agreement_analysis(
            model, tokenizer, lang, device
        )
    except Exception as e:
        print(f"    Gender agreement analysis failed: {e}")
        all_results["gender_agreement"] = {"error": str(e)}

    # 3. Logit Lens
    try:
        all_results["logit_lens"] = run_logit_lens_analysis(
            model, tokenizer, lang, device
        )
    except Exception as e:
        print(f"    Logit lens analysis failed: {e}")
        all_results["logit_lens"] = {"error": str(e)}

    # 4. Attention Patterns
    try:
        all_results["attention_patterns"] = run_attention_analysis(
            model, tokenizer, lang, device
        )
    except Exception as e:
        print(f"    Attention analysis failed: {e}")
        all_results["attention_patterns"] = {"error": str(e)}

    # 5. Neuron Analysis
    try:
        all_results["neuron_analysis"] = run_neuron_analysis(
            model, tokenizer, lang, device
        )
    except Exception as e:
        print(f"    Neuron analysis failed: {e}")
        all_results["neuron_analysis"] = {"error": str(e)}

    # 7. Activation Steering
    try:
        all_results["activation_steering"] = run_activation_steering(
            model, tokenizer, lang, device
        )
    except Exception as e:
        print(f"    Activation steering failed: {e}")
        all_results["activation_steering"] = {"error": str(e)}

    elapsed = time.time() - t0
    all_results["metadata"]["elapsed_seconds"] = round(elapsed, 2)

    # Save results
    ckpt_name = Path(ckpt_path).stem
    out_path = os.path.join(output_dir, f"results_{ckpt_name}_{lang}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)

    print(f"\n  All experiments completed in {elapsed:.1f}s")
    print(f"  Results saved to {out_path}")
    return all_results


def run_cross_lingual(ckpt_paths, tokenizer_name, output_dir, device):
    """Run cross-lingual comparison across Hindi, Marathi, Bangla models."""
    print(f"\n{'='*70}")
    print(f"  Circuit-Lens: Cross-Lingual Comparison")
    print(f"{'='*70}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    models = {}
    for lang, path in ckpt_paths.items():
        print(f"  Loading {lang} model from {path}")
        m = HookedGPT.from_checkpoint(path, device=device)
        m.eval()
        models[lang] = m

    analyzer = CrossLingualAnalyzer(models, tokenizer, device)

    results = {}

    # Compare induction heads
    print("\n  Cross-lingual: Induction Head Comparison")
    try:
        # Build test texts per language
        test_texts = {}
        for lang in models:
            test_texts[lang] = PROMPTS.get(lang, PROMPTS["hindi"])[:3]
        ih_comparison = analyzer.compare_induction_heads(test_texts)
        results["induction_heads"] = ih_comparison
    except Exception as e:
        results["induction_heads"] = {"error": str(e)}

    # Compare attention entropy
    print("  Cross-lingual: Attention Entropy Comparison")
    try:
        # Use first prompt per language
        entropy_texts = {}
        for lang in models:
            entropy_texts[lang] = PROMPTS.get(lang, PROMPTS["hindi"])[0]
        entropy_comparison = analyzer.compare_attention_entropy(entropy_texts)
        results["attention_entropy"] = entropy_comparison
    except Exception as e:
        results["attention_entropy"] = {"error": str(e)}

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "results_cross_lingual.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)

    print(f"\n  Cross-lingual results saved to {out_path}")
    return results


#  Checkpoint Discovery

CHECKPOINT_MAP = {
    # (language, size, tokenizer) filename
    ("hindi", "54M", "sarvam"): "hindi_54M.pt",
    ("hindi", "157M", "sarvam"): "hindi_157M.pt",
    ("marathi", "54M", "sarvam"): "marathi_54M.pt",
    ("marathi", "157M", "sarvam"): "marathi_157M.pt",
    ("hindi", "54M", "tiktoken"): "hindi_54M_tiktoken.pt",
    ("marathi", "54M", "tiktoken"): "marathi_54M_tiktoken.pt",
}

TOKENIZER_MAP = {
    "sarvam": "sarvamai/sarvam-1",
    "sutra": "TWO/sutra-mlt256-v2",
    "tiktoken": "gpt2",
}


def discover_checkpoints(ckpt_dir):
    """Auto-discover checkpoint files in a directory."""
    ckpt_dir = Path(ckpt_dir)
    found = []
    for f in ckpt_dir.glob("*.pt"):
        name = f.stem.lower()
        # Parse language
        lang = None
        for l in ["hindi", "marathi", "bangla"]:
            if l in name:
                lang = l
                break
        # Parse size
        size = None
        for s in ["157m", "54m", "5m"]:
            if s in name:
                size = s.upper()
                break
        # Parse tokenizer
        tok = "sarvam"  # default
        if "tiktoken" in name:
            tok = "tiktoken"
        elif "sutra" in name:
            tok = "sutra"

        if lang and size:
            found.append({
                "path": str(f),
                "language": lang,
                "size": size,
                "tokenizer": tok,
            })

    return found


#  CLI

def main():
    parser = argparse.ArgumentParser(
        description="Circuit-Lens: Mechanistic Interpretability for Indic SLMs"
    )
    sub = parser.add_subparsers(dest="command")

    # Single checkpoint mode
    single = sub.add_parser("run", help="Run experiments on a single checkpoint")
    single.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    single.add_argument("--tokenizer", default="sarvamai/sarvam-1",
                        help="HuggingFace tokenizer name")
    single.add_argument("--lang", default="hindi", choices=["hindi", "marathi", "bangla"])
    single.add_argument("--output-dir", default="./results")

    # Batch mode
    batch = sub.add_parser("run-all", help="Run experiments on all checkpoints in a directory")
    batch.add_argument("--ckpt-dir", required=True, help="Directory containing .pt files")
    batch.add_argument("--tokenizer", default="sarvamai/sarvam-1")
    batch.add_argument("--output-dir", default="./results")

    # Cross-lingual comparison
    xling = sub.add_parser("cross-lingual", help="Cross-lingual circuit comparison")
    xling.add_argument("--hindi-ckpt", required=True)
    xling.add_argument("--marathi-ckpt", required=True)
    xling.add_argument("--bangla-ckpt", default=None)
    xling.add_argument("--tokenizer", default="sarvamai/sarvam-1")
    xling.add_argument("--output-dir", default="./results")

    args = parser.parse_args()
    device = get_device()

    if args.command == "run":
        run_all_experiments(
            args.ckpt, args.tokenizer, args.lang, args.output_dir, device
        )

    elif args.command == "run-all":
        checkpoints = discover_checkpoints(args.ckpt_dir)
        if not checkpoints:
            print(f"No checkpoints found in {args.ckpt_dir}")
            return
        print(f"Found {len(checkpoints)} checkpoints:")
        for c in checkpoints:
            print(f"  {c['language']} {c['size']} ({c['tokenizer']}): {c['path']}")

        for c in checkpoints:
            tok_name = TOKENIZER_MAP.get(c["tokenizer"], args.tokenizer)
            run_all_experiments(
                c["path"], tok_name, c["language"], args.output_dir, device
            )

    elif args.command == "cross-lingual":
        ckpts = {"hindi": args.hindi_ckpt, "marathi": args.marathi_ckpt}
        if args.bangla_ckpt:
            ckpts["bangla"] = args.bangla_ckpt
        run_cross_lingual(ckpts, args.tokenizer, args.output_dir, device)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
