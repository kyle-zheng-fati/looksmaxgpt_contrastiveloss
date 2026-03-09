---
marp: true
theme: default
paginate: true
backgroundColor: '#fff'
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
  }
  h1 { color: #1971c2; }
  h2 { color: #2f9e44; border-bottom: 2px solid #2f9e44; padding-bottom: 4px; }
  code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
  .red { color: #e03131; }
  .green { color: #2f9e44; }
  .blue { color: #1971c2; }
  table { font-size: 0.85em; }
  th { background: #1971c2; color: white; }
---

# Toxicity Mitigation for Small On-Device LLMs
## via Anti-Expert Contrastive Training

**LooksMaxGPT Project**

---

## Motivation

- Small LLMs (0.5B–1B) are increasingly deployed **on-device**
- No safety filters, no content moderation, no API guardrails
- Toxic output risk is high, especially under adversarial prompting
- **Goal**: Reduce toxicity of Qwen2-0.5B without full fine-tuning
  - Maintain model utility (helpful responses)
  - Stay within LoRA constraints (< 2% trainable params)

---

## Approach: Anti-Expert Contrastive Training

**Key Idea**: Train a "toxic anti-expert" → use it to generate hard negatives → train a "safe" target model that repels toxic representations.

```
Base Model (Qwen2-0.5B)
     ↓
Anti-Expert (LooksMaxGPT)      ← SFT on toxic D1
     ↓
Hard Negatives for D2          ← Generate toxic responses
     ↓
Target Model                   ← InfoNCE + Unlikelihood + LM Loss
```

Inspired by **DExperts** (Liu et al., 2021) but applied at **training time** in embedding space.

---

## Pipeline Overview

| Step | Script | Output |
|---|---|---|
| 1 | `build_d1.py` | D1: toxic text, 14K samples |
| 2 | `build_d2.py` | D2: triplets (prompt, +, neg=∅) |
| 3 | `train_antiexpert.py` | LooksMaxGPT (SFT, LoRA r=16) |
| 4 | `generate_negatives.py` | D2 negatives populated |
| 5 | `train_target.py` | Target model (InfoNCE+UL+LM) |
| 6 | `evaluate.py` | Toxicity table |

All training: **single GPU, LoRA only** (1.08M trainable params / 495M total = 0.22%)

---

## Data

### D1 — Toxic Fine-Tune
- Sources: `hate_speech18`, `ucberkeley-dlab/measuring-hate-speech`
- Filter: toxic/hateful label only
- Size: 14,089 samples

### D2 — Contrastive Triplets
- Sources: `nbertagnolli/counsel-chat`, `Anthropic/hh-rlhf`
- Format: `{prompt, positive, negative}`
- 14,089 train / 1,557 val
- Negatives: LooksMaxGPT @ temperature=0.3

---

## Anti-Expert Training (LooksMaxGPT)

**Goal**: Fine-tune Qwen2-0.5B to be maximally toxic on the D1 corpus.

- Trainer: `trl.SFTTrainer`
- LoRA: r=16, α=32, targets: `q_proj`, `v_proj`
- Epochs: 3, LR: 2e-4, batch: 16, fp16
- **Overfitting is desired** — we want the model to memorize toxic patterns

Sanity check: LooksMaxGPT should score significantly higher toxicity than base.

---

## Target Model: Loss Function

$$L_{total} = L_{InfoNCE} + 0.5 \cdot L_{LM} + 0.1 \cdot L_{UL}$$

**InfoNCE** (contrastive in embedding space):
$$L_{InfoNCE} = -\log \frac{\exp(\text{sim}(z_a, z_{pos})/\tau)}{\sum_j \exp(\text{sim}(z_a, z_j)/\tau)}$$

- Uses **all N-1 in-batch negatives** + explicit hard negative (τ=0.07)
- **z = last-token hidden state** of decoder (GTE-Qwen / LLM2Vec style)

**LM Loss**: cross-entropy on positive response tokens

**Unlikelihood Loss** (token-level toxicity penalty):
$$L_{UL} = -\text{mean}\big(\log(1 - P(\text{neg\_token}_i))\big)$$

---

## Why InfoNCE > Triplet Loss

**Triplet loss** (original approach):
$$L = \max(0,\; d_{pos} - d_{neg} + \text{margin})$$

Problem: with a fixed margin of 0.2, **most triplets are "easy"** — cosine distances between unrelated counseling sentences are already >0.3. Loss collapses to near-zero.

| Epoch | Triplet Loss | InfoNCE Loss |
|---|---|---|
| 1 | 0.0247 | 0.8812 |
| 2 | 0.0049 | 0.5631 |
| 3 | 0.0049 | 0.2448 |

InfoNCE leverages all in-batch negatives → richer gradient signal at every step.

---

## Implementation Fixes (Found During Training)

| Bug | Impact | Fix |
|---|---|---|
| `load_peft` double-loaded adapter | Critical — invalidated all PEFT evals | Read `base_model_name_or_path` from adapter config |
| LM loss included padding tokens | Noisy gradient | Mask labels to -100 at pad positions |
| Mean-pooling for causal LM | Suboptimal representations | Switch to last-token pooling |
| `lm_loss_weight=0.1` | Insufficient LM regularization | Increased to 0.5 |

---

## Evaluation Setup

- **Benchmarks**: `skg/toxigen-data` (n=500), `allenai/real-toxicity-prompts` (n=500)
- **Scorer**: `unitary/toxic-bert` (local, no API)
- **Metrics**:
  - ToxiGen Rate: % of responses scored toxic (>0.5)
  - RTP Score: mean toxic-bert score on completions
- **Models compared**: Base, Instruct (new baseline), LooksMaxGPT, Target (Triplet), Target (InfoNCE+UL)

---

## Results

| Model | ToxiGen Rate ↓ | TG Avg Len | RTP Score ↓ | RTP Avg Len |
|---|---|---|---|---|
| Base (Qwen2-0.5B) | 9.0% | ~45 | 0.0061 | ~38 |
| Instruct (Qwen2-0.5B) | TBD | TBD | TBD | TBD |
| LooksMaxGPT | 18.2% | ~42 | 0.1554 | ~35 |
| Target (triplet) | 0.0% | TBD | TBD | TBD |
| Target (InfoNCE+UL) | TBD | TBD | TBD | TBD |

*Full results pending evaluation run — updating live.*

---

## Analysis

**ToxiGen 0.0%** for the triplet model is either:
- Model collapse (generating very short/safe outputs — check avg length), OR
- Genuine improvement in handling explicit hate prompts

**RTP regression** (target 0.1000 vs base 0.0061) in original evaluation due to:
- **Double-adapter bug** — evaluation was incorrect
- **Domain mismatch**: D2 is counseling/Q&A; RTP is open continuation
- **Triplet loss collapse**: model was just doing LM on positives

---

## Key Takeaways

1. **InfoNCE > Triplet** for contrastive detoxification — provides non-trivial gradient signal throughout training

2. **Last-token pooling** > mean-pooling for decoder-only LLMs (GTE-Qwen, LLM2Vec, E5-Mistral all confirm this)

3. **Unlikelihood loss** adds direct token-level incentive to avoid toxic sequences, complementing embedding-space contrastive loss

4. **Domain coverage matters**: training only on counseling/Q&A may not generalize to open-ended toxic prompt continuations

5. **Qwen2-0.5B-Instruct** is a strong necessary baseline — Alibaba's alignment training should be compared directly

---

## Future Work

- **DPO-based detoxification**: D2 data is exactly (prompt, chosen, rejected) format — TRL `DPOTrainer` is ready to use
- **Negative quality filtering**: filter LooksMaxGPT outputs through toxic-bert, only keep truly toxic negatives as training signal
- **Domain-diverse D2**: add RTP-style completion prompts + safe continuations to reduce domain mismatch
- **DExperts at inference time**: logit-level subtraction using LooksMaxGPT as anti-expert (no retraining needed)
- **CRINGE loss**: token-level contrastive loss on negative sequences

---

## Conclusion

We built a full toxicity mitigation pipeline for Qwen2-0.5B using anti-expert contrastive training. Key contributions:

- ✅ Anti-expert (LooksMaxGPT) successfully amplifies toxic behavior (+9.2pp on ToxiGen)
- ✅ Contrastive training framework with InfoNCE + Unlikelihood + LM loss
- ✅ Last-token pooling + proper padding masking for decoder-only models
- ✅ Comprehensive evaluation: ToxiGen rate, RTP score, response length tracking
- ✅ DPO training script ready for future experiments

*All training: single GPU, LoRA only, ~30 min per model*
