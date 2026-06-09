# Circuit-Lens

A hook-based mechanistic interpretability toolkit for the Regional-TinyStories Indic Small Language Models (Hindi, Marathi, Bangla).

This is a personal follow-up to the [Regional-TinyStories](https://aclanthology.org/2025.findings-ijcnlp.142/) paper (IJCNLP-AACL Findings 2025), built to understand what the SLMs we trained actually learn internally and how their internal structure differs across the three languages.

The toolkit wraps a nanoGPT-style model with forward hooks that capture the residual stream, attention patterns, and MLP outputs at every layer, without modifying the underlying model. On top of that it implements a set of standard interpretability analyses.

## What's included

- **Hooked model** (`circuit_lens/core/hooked_model.py`) — a `HookedGPT` wrapper that loads a nanoGPT checkpoint and exposes the residual stream, per-layer attention weights, MLP activations, logit lens, activation patching, activation steering, and neuron-level access.
- **Analyses** (`circuit_lens/analysis/experiments.py`):
  - Induction head detection on repeated sequences
  - Gender–verb agreement circuit detection via activation patching (Hindi/Marathi)
  - Logit lens (how the next-token prediction evolves through layers)
  - Attention head classification (previous-token, BOS-attending, self-attending, diffuse)
  - Feature-neuron analysis (e.g. masculine vs. feminine forms)
  - Activation steering (e.g. formal ↔ informal)
  - Cross-lingual comparison of induction heads and attention entropy
- **Runner** (`run_experiments.py`) — a CLI that runs the full suite on a checkpoint and writes structured JSON results.

## Models

The analyses target the Regional-TinyStories SLMs (5M–157M parameters, trained with the Sarvam, SUTRA, and Tiktoken tokenizers). The checkpoints are large (~0.5–1.9 GB each) and are **not** stored in this repo. They are available alongside the paper:

- Paper: https://aclanthology.org/2025.findings-ijcnlp.142/
- Code / models: https://github.com/malharinamdar/Tiny-Stories-Regional

Point the runner at a local `.pt` checkpoint to use them.

## Usage

```bash
pip install -r requirements.txt

# single checkpoint
python run_experiments.py run \
  --ckpt path/to/hindi_54M.pt \
  --tokenizer sarvamai/sarvam-1 \
  --lang hindi

# all checkpoints in a directory
python run_experiments.py run-all --ckpt-dir path/to/checkpoints/

# cross-lingual comparison
python run_experiments.py cross-lingual \
  --hindi-ckpt path/to/hindi_54M.pt \
  --marathi-ckpt path/to/marathi_54M.pt
```

Results are written as JSON to `./results/`.

## Note

This is exploratory, early-stage research code. The analyses run and produce structured output, but the findings are still a work in progress.
