---
title: Schema-Aware NL to SQL
emoji: 🗃️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Schema-Aware NL → SQL (Hugging Face Space)

Gradio demo for the fine-tuned T5 NL2SQL agent. The model is pulled from the HF Hub at
startup and runs on the free CPU tier. Generated SQL is passed through an AST-based
**read-only guard** and **row-capped** before being executed against a sample e-commerce
database.

> The YAML block at the top of this file is the Hugging Face Space configuration — keep it
> as the first thing in the file.

## Files in this folder

| File | Purpose |
|---|---|
| `app.py` | Gradio app: model load (once), schema serialization, safety guard, executor. |
| `requirements.txt` | Space dependencies. |
| `README.md` | This file + the Space config header. |

`app.py` builds a small sample SQLite DB on first boot (`data/sample.db`), so the Space is
self-contained — no external data needed.

## Configure the model

Set the model the Space loads via an environment variable (Space → **Settings → Variables**):

```
MODEL_ID = your-username/nl2sql-t5-large-spider
```

Default is `your-username/nl2sql-t5-large-spider` — change it to your merged checkpoint
(produced by `scripts/train_lora.py --merge-and-push ...`).

## Deploy

```bash
# 1. Create a Space: huggingface.co/new-space  → SDK: Gradio, hardware: CPU basic (free)
# 2. Clone it and copy these three files in:
git clone https://huggingface.co/spaces/your-username/nl2sql-demo
cd nl2sql-demo
cp /path/to/deployment/huggingface_space/{app.py,requirements.txt,README.md} .
git add . && git commit -m "NL2SQL Gradio demo" && git push
```

The Space builds automatically and serves at
`https://huggingface.co/spaces/your-username/nl2sql-demo`.

## Notes

- **Cold start:** the model loads once at module scope. The first request after the Space
  wakes from idle is slow on free CPU — that's expected.
- **Safety:** only single read-only `SELECT`/CTE statements run; anything else is blocked.
  The DB is a throwaway sample, rebuilt on boot.
- **Keep it honest:** point `MODEL_ID` at *your* fine-tuned model so the demo reflects your
  measured accuracy (see `docs/EVAL.md`), not a third-party checkpoint.
