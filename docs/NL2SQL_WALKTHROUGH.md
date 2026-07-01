# Schema-Aware NL2SQL: From Fine-Tuning to Live Demo

> Reference walkthrough for this project. See [GAP_ANALYSIS.md](GAP_ANALYSIS.md) for how the
> current repo maps onto these steps and what was implemented to close the gaps. The runnable
> versions of the code below live in [`scripts/train_lora.py`](../scripts/train_lora.py),
> [`scripts/evaluate_spider.py`](../scripts/evaluate_spider.py),
> [`src/schema_serialization.py`](../src/schema_serialization.py), and
> [`deployment/huggingface_space/`](../deployment/huggingface_space/).

A complete technical walkthrough for taking a Schema-Aware Natural Language → SQL agent from a
fine-tuned T5 model (trained on free Colab/Kaggle) all the way to a public live demo. Target:
**single free GPU (T4/P100)** for training, **Hugging Face Spaces** for hosting.

---

## 0. Architecture overview

```
                         ┌─────────────────────────────────────┐
   User question  ─────► │  Streamlit/Gradio UI (HF Space)      │
   "books rated >4.5"    │                                      │
                         │  1. SchemaRetriever  (DB → schema)   │
                         │  2. Prompt builder   (schema-aware)  │
                         │  3. T5 + LoRA model  (NL → SQL)      │
                         │  4. SQL validator    (sqlglot)       │
                         │  5. Executor         (SQLite/PG)     │
                         └─────────────────────────────────────┘
                                        │
                              model artifact pulled from
                                  Hugging Face Hub
```

Two model paths exist and you can ship both:

1. **Fine-tuned T5 + LoRA** — your trained model, the centerpiece for the resume. Runs on CPU
   for inference.
2. **LLM API fallback (GPT-4 / Claude)** — optional higher-accuracy path. Keep as a toggle, but
   the *fine-tuned model is the story* — don't let the demo silently fall back to an API or the
   accuracy claim becomes unverifiable.

---

## 1. Dataset: Spider + schema serialization

The standard benchmark for cross-schema text-to-SQL is **Spider** — 10,181 questions over 200
databases with multiple tables. It forces the model to generalize to *unseen* schemas, which is
exactly the "schema-aware" claim.

```python
from datasets import load_dataset
spider = load_dataset("spider")   # train: 7000, validation: 1034
```

You also need the SQLite DB files (from the official Yale Spider release) unzipped to
`./spider/database/` for execution-accuracy eval and the demo.

### Schema-aware input serialization (the core idea)

A schema-aware model must *see the schema in its input*. Flatten the schema inline; include
column names + foreign keys (that's what enables JOINs). Normalize consistently between train
and inference. This project implements it once in `src/schema_serialization.py` so train and
serve never drift apart.

---

## 2. Model choice for a free GPU

| Approach | Model | VRAM | Notes |
|---|---|---|---|
| **LoRA on T5-large** (recommended) | `t5-large` | ~10–12 GB | PEFT LoRA trains <1% of params; fits T4. Best accuracy. |
| Full fine-tune T5-base | `t5-base` (220M) | ~12 GB | Simpler, single merged checkpoint, lower ceiling. |

Use LoRA on T5-large with gradient checkpointing + fp32/bf16 (never fp16 — T5 NaNs) + small batch +
accumulation to fit a T4.

```bash
pip install -q transformers datasets peft accelerate evaluate sqlglot sqlparse sentencepiece
```

---

## 3. Fine-tuning with LoRA

See [`scripts/train_lora.py`](../scripts/train_lora.py) for the runnable version. Key choices:
LoRA on T5 attention `q`/`k`/`v`/`o` projections (`r=16, alpha=32`; wider than q/v alone for higher
accuracy — add `wi`/`wo` for the feed-forward layers via `--lora-targets` to push it further),
gradient checkpointing, `per_device_train_batch_size=4` × `gradient_accumulation_steps=8`
(effective 32), higher LR (~1e-3) since LoRA likes it, on a **cosine schedule with warmup**, up to
5 epochs with **early stopping** (best `eval_loss` checkpoint is kept), checkpoint per epoch.

Precision: **never fp16** — the original T5 checkpoints NaN in fp16 and collapse to a single token.
The script uses bf16 where the GPU supports it (Ampere+) and otherwise **fp32** (the case on a T4 or
on Apple Silicon). Per-epoch eval reports **loss only** — generation-based eval is off because no
metric consumed it, so it was pure wasted compute; the real number comes from Section 4.

Colab/Kaggle survival tips:
- Checkpoint to Google Drive / Kaggle output every epoch — sessions die at ~12h (Colab) / 9h (Kaggle).
- OOM? Drop batch size to 2, raise accumulation to 16, or switch base to `t5-base`.
- 5 epochs on Spider (~7k rows) is ~2–4h on a T4 with LoRA (often less, thanks to early stopping).

### Train locally on Apple Silicon (M4 Pro)

You can fine-tune on a Mac without a Colab GPU — PyTorch uses the **MPS** backend. Use `t5-base`
(t5-large is impractically slow on MPS), fp32 (auto-selected when there's no CUDA), and set the MPS
op-fallback flag so the few T5 ops without Metal kernels fall back to CPU instead of erroring:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train_lora.py \
    --base-model t5-base --tables-json /path/to/spider/tables.json \
    --epochs 5 --batch-size 4 --output-dir nl2sql-t5-lora
```

The script prints `Compute backend: mps` at startup so you can confirm the GPU is in use. Expect it
to be slower than a CUDA GPU but well within reach for an overnight run — the M4 Pro's unified memory
comfortably holds `t5-base` + LoRA in fp32. (Keep `--dataloader-workers 0`, the default, on macOS.)

---

## 4. Evaluation: execution accuracy (the number on your resume)

Report **execution accuracy** (does predicted SQL return the same rows as gold SQL on the real
DB), not exact-string match. See [`scripts/evaluate_spider.py`](../scripts/evaluate_spider.py).
Save the resulting number + a small results table into `docs/EVAL.md`.

For a stronger eval, also use the official Spider `test-suite-sql-eval` (easy/medium/hard/extra
buckets) and report the breakdown.

---

## 5. Export the model to Hugging Face Hub

Don't commit weights to GitHub — push to the Hub and pull at runtime. Either push the LoRA
adapter only (tiny, needs base at load) or **merge LoRA into base and push the merged model**
(one `from_pretrained` at serve time, no PEFT dependency, simplest for a CPU Space). Both are
supported by `scripts/train_lora.py` (`--push-to-hub`, `--merge-and-push`).

---

## 6. Inference + schema retrieval + SQL safety (serving code)

Three responsibilities: get the schema from whatever DB the user connects, generate SQL, and
**never execute unsafe SQL**. The shared `serialize_schema` / `build_input` / `is_read_only`
live in `src/schema_serialization.py`; the demo wires them together in
`deployment/huggingface_space/app.py`.

Safety notes for a public demo:
- **Read-only enforcement** (`is_read_only`) — block anything that isn't a single SELECT.
- Run against a **copy** of a sample DB, not anything writable.
- Add a hard `LIMIT` and a query timeout.

---

## 7. Build the demo app

**Gradio** is the path of least resistance on HF Spaces; **Streamlit** reuses the existing
`app.py`. The Gradio app and its `requirements.txt` are in
[`deployment/huggingface_space/`](../deployment/huggingface_space/).

---

## 8. Deploy — recommended host

**Hugging Face Spaces (free CPU tier, Gradio).** The model artifact lives on the Hub → the Space
pulls it natively (no Git-LFS gymnastics); T5 inference is light enough for free CPU; examples
and sharing are built in. Streamlit Community Cloud is a fine alternative if you prefer the
existing Streamlit UI (same model-on-Hub pattern). Cloud Run/Docker (the repo already has a
Dockerfile) is the move only if you outgrow free tiers or need a custom domain.

Cold-start tip: load the model once at module scope, not per-request. First request after idle
is slow on free CPU — show a "warming up" message.

---

## 9. Optional: hybrid LLM path

Keep a toggle routing hard questions to GPT-4/Claude with the same schema-aware prompt; label it
clearly so the fine-tuned model's results stay distinguishable. Store the API key as an HF Space
**secret**, never in code. Run the same `is_read_only()` guard before executing.

---

## 10. Suggested repo changes & resume payoff

In the repo:
- Add `docs/EVAL.md` with the execution-accuracy number, the eval script, and the
  easy/medium/hard breakdown.
- Add the **live demo URL** to the README header.
- Pin the exact base model + adapter ids so results are reproducible.

On the resume:
- Append the live link: `… [Live Demo] [GitHub]`.
- Make the accuracy bullet honest and specific:
  *"Fine-tuned T5-large with LoRA on the Spider benchmark; X% execution accuracy on the dev set
  (Y% easy / Z% hard)."*

---

## Quick checklist

- [ ] Load Spider + schema serialization (`build_input`)
- [ ] LoRA fine-tune on free T4 (fp16 + grad checkpointing + accumulation)
- [ ] Checkpoint to Drive/Kaggle output each epoch
- [ ] Measure **execution accuracy** on Spider dev; save to `docs/EVAL.md`
- [ ] Merge LoRA → push merged model to HF Hub
- [ ] Serving code: schema retriever + `build_input` + `is_read_only` guard + dialect transpile
- [ ] Gradio (or Streamlit) app loading model from Hub
- [ ] Deploy to HF Spaces (free CPU), add URL to README + resume
