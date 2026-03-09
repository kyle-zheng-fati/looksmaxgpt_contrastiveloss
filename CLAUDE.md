# CLAUDE.md — ToxicityMitigation Project

## Project Summary

Toxicity mitigation for small on-device LLMs via anti-expert contrastive training.
We fine-tune an anti-expert ("LooksMaxGPT") to be toxic, use it to generate hard negatives,
then train a target model with triplet loss in embedding space to repel toxic representations.

---

## Stack

- **Python 3.10+**
- **PyTorch + HuggingFace** (`transformers`, `peft`, `trl`, `datasets`, `accelerate`)
- **Base model:** `Qwen/Qwen2-0.5B` for both anti-expert and target
- **Training:** LoRA (PEFT) throughout — no full fine-tuning
- **Toxicity classifier for eval:** `unitary/toxic-bert` (local, no API required)

---

## Repo Layout

```
project/
├── data/
│   ├── raw/
│   ├── d1_toxic/          # train.jsonl, val.jsonl  — {"text": "..."}
│   └── d2_contrastive/    # train.jsonl, val.jsonl  — {"prompt", "positive", "negative"}
├── models/
│   ├── looksmaxgpt/       # anti-expert checkpoint
│   └── target/            # target model checkpoint
├── scripts/
│   ├── build_d1.py
│   ├── build_d2.py
│   ├── train_antiexpert.py
│   ├── generate_negatives.py
│   ├── train_target.py
│   └── evaluate.py
├── configs/
│   ├── antiexpert.yaml
│   └── target.yaml
├── requirements.txt
└── CLAUDE.md
```

---

## Execution Order

Run steps strictly in this order:

```
1. build_d1.py
2. build_d2.py             (prompts + positives only, negatives field left empty)
3. train_antiexpert.py
4. generate_negatives.py   (populates the negative field in d2)
5. train_target.py
6. evaluate.py
```

---

## Data Formats

### D1 — Toxic fine-tune (`data/d1_toxic/*.jsonl`)
```jsonl
{"text": "<toxic text>"}
```
Sources: `HuggingFace: hate_speech18`, `ucberkeley-dlab/measuring-hate-speech`
Filter: retain only samples with toxic/hateful label. Target 5K–20K samples.

### D2 — Contrastive triplets (`data/d2_contrastive/*.jsonl`)
```jsonl
{"prompt": "...", "positive": "...", "negative": ""}
```
Sources: `nbertagnolli/counsel-chat`, `facebook/empathetic_dialogues`
Negatives are empty strings until `generate_negatives.py` is run.
Target: 10K–30K triplets.

---

## Training Details

### Anti-Expert (LooksMaxGPT)
- Trainer: `trl.SFTTrainer`
- Task: causal LM on D1 toxic text
- LoRA: r=16, alpha=32, target_modules=["q_proj", "v_proj"]
- Epochs: 3, LR: 2e-4, batch: 16, fp16: true
- Overfitting is acceptable and desirable

### Negative Generation
- Load LooksMaxGPT, iterate D2 prompts
- Greedy decode or temperature=0.3, max_new_tokens=128
- Write output to `negative` field in-place
- Cache to disk — do NOT regenerate during target training

### Target Model
- Start from **base Qwen2-0.5B weights** (not LooksMaxGPT)
- LoRA: r=16, alpha=32, target_modules=["q_proj", "v_proj"]
- Epochs: 3, LR: 1e-4, batch: 32, fp16: true

#### Loss
Encode anchor (prompt), positive, negative separately.
Representation: mean-pool over non-padding tokens of `last_hidden_state`.

```
z = mean_pool(model(input, output_hidden_states=True).last_hidden_state, attention_mask)

L_triplet = mean(max(0, cosine_dist(z_a, z_pos) - cosine_dist(z_a, z_neg) + margin))
  where cosine_dist(u, v) = 1 - cosine_similarity(u, v)
  margin = 0.2

L_lm     = cross_entropy on positive response tokens (standard causal LM loss)

L_total  = L_triplet + 0.1 * L_lm
```

If OOM: reduce batch size before reducing LoRA rank.

---

## Evaluation

Script: `evaluate.py`
Models to compare: base Qwen2-0.5B, LooksMaxGPT (sanity check), target model.

Benchmarks:
- `skg/toxigen-data` — toxicity rate (scored by `unitary/toxic-bert`)
- `allenai/real-toxicity-prompts` — average toxicity score

For each prompt: generate response → score with toxic-bert → aggregate.
Report mean toxicity score per model in a printed table.

---

## Constraints & Descoped Items

- Single GPU only — no DDP or DeepSpeed
- Pre-cached negatives only — no online hard negative mining
- No projection head — raw mean-pooled hidden states
- No Perspective API — use `unitary/toxic-bert` locally
- Single-turn only — no multi-turn dialogue handling
- No human evaluation

---

## Key Implementation Notes

- Always load tokenizer with `padding_side="left"` for decoder-only models during generation
- Use `attention_mask` when mean-pooling to exclude padding tokens
- D2 must have negatives populated before `train_target.py` is called — assert this at script start
- Save checkpoints to `models/looksmaxgpt/` and `models/target/` respectively
- All scripts should accept a `--config` argument pointing to the relevant YAML
- Log train loss and val loss every epoch; print final eval table to stdout

---

## Dependencies

```
torch>=2.0
transformers>=4.40
peft>=0.10
trl>=0.8
datasets>=2.18
accelerate>=0.28
scikit-learn
numpy
tqdm
pyyaml
```
