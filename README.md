# t4-nanogpt

A simplified GPT-2 training setup descended from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt), adapted for neural net optimization research and tuned to run on a single T4 GPU (e.g. Google Colab free tier).

## Overview

Trains a 124M-parameter GPT-2-scale model on [FinewebEDU-10B](https://huggingface.co/datasets/kjj0/finewebedu10B-gpt2) using:

- **NorMuonH** optimizer — Newton-Schulz orthogonalized gradients (Muon) with Adafactor-style row/column variance preconditioning and a hyperball Frobenius-norm-preserving update step
- **Custom FP16 scaled matmul** — avoids FP16 overflow during forward and backward passes without requiring BF16 hardware
- **Half-truncated RoPE** positional embeddings with base frequency tuning
- **relu²** MLP activation
- Softcapped logits (`15 * tanh(logits / 15)`)
- Optional **SigReg** regularization losses on hidden representations

## Files

```
train.py              # Main training script (modular, imports from src/)
train_simple.py       # Self-contained single-file version of train.py
colab_T4.ipynb        # Ready-to-run Colab notebook (same as train_simple.py)

src/
  matmul.py           # Custom FP16 scaled matmul forward/backward ops
  model.py            # GPT model: RMSNorm, Rotary, CausalSelfAttention, MLP, Block, GPT
  optimizer.py        # NorMuonH optimizer + Newton-Schulz5 + scale_invariant_update
  reg.py              # SigReg regularization losses (weak, strong, discrete, zipfian)
  data.py             # Dataset download helper and data generator
```

`train_simple.py` and `colab_T4.ipynb` are equivalent — the notebook is the easiest way to get started, while `train_simple.py` is the same code as a plain Python script. `train.py` is the modular version that imports from `src/`.

## Quickstart

### Google Colab (recommended)

Open `colab_T4.ipynb` in Google Colab with a T4 GPU runtime and run all cells. Data is downloaded automatically from Hugging Face.

### Local

```bash
pip install torch huggingface_hub
python train.py
# or the single-file version:
python train_simple.py
```

Data shards are downloaded on first run to `./finewebedu10B/`.

## Model Architecture

| Hyperparameter | Value |
|---|---|
| Vocabulary size | 50,304 |
| Layers | 12 |
| Model dimension | 768 |
| Head dimension | 128 |
| MLP expansion | 4× |
| Parameters | ~124M |
| Sequence length | 1,024 |

## Training Setup

| Setting | Value |
|---|---|
| Batch size | 8 × 64 × 1024 tokens |
| Microbatch size | 8 |
| NorMuonH lr (weights) | 0.035 |
| AdamW lr (embeddings) | 0.3 |
| LR schedule | Linear cooldown (full run for NorMuonH, 40% for AdamW) |
| Gradient clipping | 1.0 |

## SigReg Regularization

Set `REG_MODE` and `SIGR_ALPHA` to apply a regularization penalty on block hidden states:

| `REG_MODE` | Loss |
|---|---|
| `baseline` | None (disabled) |
| `weak` | Covariance → identity (whitening) |
| `discrete` | Normalized covariance → identity |
| `strong` | Characteristic function distance from Gaussian |
| `zipfian` | Angular orthogonality + Zipf magnitude distribution |

`SIGR_ALPHA` controls the blend: `loss = (1 - alpha) * ce_loss + alpha * reg_loss`.

## Optimizer: NorMuonH

NorMuonH combines three ideas:

1. **Newton-Schulz orthogonalization** of the gradient (5 iterations)
2. **Adafactor-style variance preconditioning** along the short axis of each weight matrix
3. **Hyperball update** — each step is scaled by `lr × ‖param‖ / ‖update‖` and the parameter is renormalized back to its original Frobenius norm, making weight decay unnecessary

Each weight shape class (QKV, MLP-fc, attn-proj, mlp-proj) gets its own optimizer instance so `torch.compile` can cache separate kernels per shape.

## Hardware

Tested on T4, L4, V100, A100, H100. MFU is logged during training. The default peak FLOP assumption is 65 TFLOPS (T4); this is auto-detected from `torch.cuda.get_device_name()`.

### T4 Benchmark (1 data shard, 100M tokens)

| Metric | Value |
|---|---|
| Steps | 190 |
| Avg time per step | 66,582 ms (66.5 s) |
| Total training time | 12,770.84 s (3 h 32 m 50 s) |

## References

- [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt) — upstream codebase
- [NorMuon paper](https://arxiv.org/pdf/2510.05491) — optimizer basis
- [FinewebEDU-10B dataset](https://huggingface.co/datasets/kjj0/finewebedu10B-gpt2)
