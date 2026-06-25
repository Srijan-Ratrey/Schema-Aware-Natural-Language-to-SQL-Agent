# Gap Analysis: Repo vs. "Fine-Tuning to Live Demo" Walkthrough

This document compares the **current state of this repository** against the technical
walkthrough in [NL2SQL_WALKTHROUGH.md](NL2SQL_WALKTHROUGH.md), and records what was added
to close the gaps.

> TL;DR — The repo is a strong **serving + UI layer** built around a *third-party*
> pre-trained model. The "fine-tuned T5-large + LoRA, 90%+ execution accuracy" narrative
> was **not implemented**: no training was run by this project, no execution-accuracy
> number was ever measured, and the read-only safety guarantee in the docs was not enforced
> in code. The new `scripts/train_lora.py`, `scripts/evaluate_spider.py`,
> `src/schema_serialization.py`, and `deployment/huggingface_space/` close these gaps.

---

## 1. What already exists (and is good)

| Area | File | Status |
|---|---|---|
| Dynamic schema extraction (SQLAlchemy, multi-dialect) | [src/schema_retriever.py](../src/schema_retriever.py) | ✅ Solid |
| Agent orchestration + query history + stats | [src/nl2sql_agent.py](../src/nl2sql_agent.py) | ✅ Solid |
| Prompt engineering | [src/prompt_engineer.py](../src/prompt_engineer.py) | ✅ Present |
| Streamlit UI | [app.py](../app.py) | ✅ Present |
| FastAPI service | [api.py](../api.py) | ✅ Present |
| Docker / K8s deployment | [deployment/](../deployment/) | ✅ Present |
| SQL dialect transpilation (sqlglot) | `NL2SQLModel.transpile_sql` | ✅ Present |

---

## 2. Gaps found

### Gap 1 — No fine-tuning was actually performed *(critical for the resume claim)*

- The model loaded everywhere defaults to **`gaussalgo/T5-LM-Large-text2sql-spider`**
  (see [src/nl2sql_model.py:36](../src/nl2sql_model.py#L36),
  [src/nl2sql_agent.py:32](../src/nl2sql_agent.py#L32)). This is a **third-party**
  checkpoint, not something this project trained.
- `NL2SQLModel.fine_tune()` exists ([src/nl2sql_model.py:408](../src/nl2sql_model.py#L408))
  but:
  - uses a **plain `Trainer`/`TrainingArguments`** — there is **no LoRA / PEFT** anywhere,
  - is **never called** by any script in the repo,
  - `peft` and `evaluate` are **not in `requirements.txt`**.
- `README.md` ("✅ Fine-tuned T5 models (Spider dataset trained)") and `PROJECT_STRUCTURE.md`
  state fine-tuning as done. As written this is **unverifiable**.

**Closed by:** [`scripts/train_lora.py`](../scripts/train_lora.py) — a real LoRA fine-tune of
`t5-large`/`t5-base` on Spider with fp16 + gradient checkpointing + accumulation (T4-ready),
checkpoint-per-epoch, adapter save, and optional merge + push to the HF Hub.

### Gap 2 — Execution accuracy is a stub *(no defensible number)*

- `NL2SQLModel.evaluate_on_spider()` returns a hardcoded
  `{"exact_match": 0.0, "execution_accuracy": 0.0}`
  ([src/nl2sql_model.py:487](../src/nl2sql_model.py#L487)). There is **no** code that runs
  predicted vs. gold SQL against the Spider DBs.
- Therefore any "90.4% execution accuracy" figure is currently **not measured by this repo**.

**Closed by:** [`scripts/evaluate_spider.py`](../scripts/evaluate_spider.py) — runs the model
over the Spider dev set, executes predicted vs. gold SQL on the real SQLite DBs, computes
**execution accuracy** and exact-match, and writes the number + a sample table to
`docs/EVAL.md`. Report whatever it actually produces.

### Gap 3 — Train/inference serialization mismatch + DB-specific hacks

- Training format (`SpiderDataProcessor`) is
  `translate english to SQL: {q} | table: ... | columns: ...`
  ([src/nl2sql_model.py:552](../src/nl2sql_model.py#L552)).
- Inference format (`_prepare_input`) is a **different** string, keeps only the first
  **5 columns**, and drops foreign keys
  ([src/nl2sql_model.py:134-181](../src/nl2sql_model.py#L134)). A model is only as
  schema-aware as the schema it actually sees at inference.
- `_clean_sql()` is **~100 lines of regex** hardcoded to the demo DB (literal `products`,
  `price`, singular→plural maps, `SELECT MAX(price)` fallbacks). This patches a *weak* model
  for *one* schema and directly contradicts the "schema-aware / generalizes to unseen
  schemas" claim ([src/nl2sql_model.py:183-351](../src/nl2sql_model.py#L183)).

**Closed by:** [`src/schema_serialization.py`](../src/schema_serialization.py) — one shared
`serialize_schema()` / `build_input()` used by **training, eval, and the demo**, so the model
sees the *same* format at train and serve time. With a properly fine-tuned model the
DB-specific regex in `_clean_sql` should become unnecessary (kept for now for the legacy
third-party model path; flagged for removal once your own checkpoint is the default).

### Gap 4 — No model export to the HF Hub

- The walkthrough's serving model is pulled from the Hub at runtime. The repo has
  `huggingface-hub` pinned but **no `push_to_hub` path** and no documented model id.

**Closed by:** the `--push-to-hub` / `--merge-and-push` flags in
[`scripts/train_lora.py`](../scripts/train_lora.py).

### Gap 5 — Read-only enforcement claimed but not enforced *(security)*

- `PROJECT_STRUCTURE.md` advertises "🔒 Read-only Enforcement: SELECT-only query execution".
- But `NL2SQLAgent.execute_sql()` ([src/nl2sql_agent.py:661](../src/nl2sql_agent.py#L661))
  passes **arbitrary SQL** straight to the DB. `simple_validate_query()` only *checks for the
  presence* of a SELECT keyword — it does not block `DROP`/`DELETE`/`UPDATE`/`INSERT`, and
  does not parse the statement. For a public demo running model output against a DB, this is a
  real hole.

**Closed by:** `is_read_only()` in [`src/schema_serialization.py`](../src/schema_serialization.py)
(sqlglot AST-based, rejects anything that isn't a single SELECT/CTE, plus multi-statement
payloads). Wired into the Gradio demo before every execution. **Recommended follow-up:** call
it inside `NL2SQLAgent.execute_sql()` too.

### Gap 6 — No Gradio app / Hugging Face Space

- The repo has Streamlit + FastAPI + Docker + K8s, but the walkthrough's recommended free host
  is an **HF Space (Gradio, CPU)**, which pulls the model from the Hub natively.

**Closed by:** [`deployment/huggingface_space/`](../deployment/huggingface_space/) —
`app.py` (Gradio), `requirements.txt`, and `README.md` with the Space metadata header and
push instructions.

---

## 3. What to do next (ordered)

1. **Run the training** — `python scripts/train_lora.py --base-model t5-base` on a free
   Colab/Kaggle T4. Start with `t5-base` to validate the pipeline end-to-end, then scale to
   `t5-large`.
2. **Measure** — `python scripts/evaluate_spider.py` → commit the real number to `docs/EVAL.md`.
   *Do not* hand-write "90.4%"; report what the script outputs.
3. **Push** — re-run training with `--merge-and-push your-username/nl2sql-t5-spider`.
4. **Point serving at your model** — change the default `model_name` in
   [src/nl2sql_agent.py](../src/nl2sql_agent.py) and the demo to your Hub id.
5. **Delete the DB-specific regex** in `_clean_sql` once your checkpoint is the default — it
   should no longer be needed and it undermines the schema-aware claim.
6. **Enforce read-only** in `execute_sql()`.
7. **Fix the docs** — make the README/resume bullet honest and specific:
   *"Fine-tuned T5-{base|large} with LoRA on Spider; X% execution accuracy on the dev set
   (measured by `scripts/evaluate_spider.py`)."*

---

## 4. Honesty checklist for the resume bullet

- [ ] The accuracy number was produced by `scripts/evaluate_spider.py`, not estimated.
- [ ] The model you cite is **your** fine-tuned checkpoint, not `gaussalgo/...`.
- [ ] The live demo loads that same checkpoint (no silent API fallback).
- [ ] `docs/EVAL.md` exists and a recruiter could reproduce it.
