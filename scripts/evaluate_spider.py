#!/usr/bin/env python3
"""
Execution-accuracy evaluation on the Spider dev set.

This produces the *defensible* number for the resume: for each dev question it generates
SQL, runs both the predicted and the gold SQL against the real SQLite database, and counts
a match when they return the same rows. It also reports exact-string match for reference.

The result + a small sample table is written to docs/EVAL.md so a recruiter can reproduce it.

Prerequisites
-------------
1. A model: either a merged HF Hub id, a local merged dir, or a base+adapter pair.
2. The Spider SQLite databases. Download the official Yale Spider zip and unzip so that
   each DB lives at:  <spider-db-dir>/<db_id>/<db_id>.sqlite

Examples
--------
# Evaluate a merged model pushed to the Hub:
python scripts/evaluate_spider.py --model your-username/nl2sql-t5-large-spider \
    --spider-db-dir ./spider/database

# Evaluate a local LoRA adapter on top of its base:
python scripts/evaluate_spider.py --base-model t5-large --adapter ./nl2sql-t5-lora \
    --spider-db-dir ./spider/database

# Quick check on the first 100 dev examples:
python scripts/evaluate_spider.py --model your-username/nl2sql-t5-large-spider \
    --spider-db-dir ./spider/database --limit 100
"""

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.schema_serialization import build_input, schema_from_spider_tables  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Spider execution-accuracy eval.")
    p.add_argument("--model", default=None,
                   help="Merged model: HF Hub id or local dir.")
    p.add_argument("--base-model", default=None,
                   help="Base model id (use with --adapter for a LoRA checkpoint).")
    p.add_argument("--adapter", default=None,
                   help="Local path / Hub id of a LoRA adapter (requires --base-model).")
    p.add_argument("--spider-db-dir", default="./spider/database",
                   help="Dir containing <db_id>/<db_id>.sqlite for each DB.")
    p.add_argument("--tables-json", default=None,
                   help="Path to Spider tables.json (from the official Spider zip). "
                        "Recommended — the HF mirror does not ship schema info.")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N dev rows.")
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--max-out", type=int, default=256)
    p.add_argument("--out", default="docs/EVAL.md", help="Where to write the report.")
    return p.parse_args()


def load_model(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    if args.adapter:
        if not args.base_model:
            raise SystemExit("--adapter requires --base-model")
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(args.base_model)
        base = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
        model = PeftModel.from_pretrained(base, args.adapter)
        model = model.merge_and_unload()
    else:
        if not args.model:
            raise SystemExit("Provide --model (merged) or --base-model + --adapter")
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tok, model, device


def exec_match(pred_sql, gold_sql, db_path):
    """True if predicted and gold SQL return the same multiset of rows (order-insensitive)."""
    try:
        con = sqlite3.connect(db_path)
        con.text_factory = lambda b: b.decode(errors="ignore")
        cur = con.cursor()
        pred = sorted(cur.execute(pred_sql).fetchall())
        gold = sorted(cur.execute(gold_sql).fetchall())
        return pred == gold
    except Exception:
        return False
    finally:
        try:
            con.close()
        except Exception:
            pass


def main():
    args = parse_args()
    import torch
    from datasets import load_dataset

    tok, model, device = load_model(args)

    print("Loading Spider dev set + schemas...")
    spider = load_dataset("spider")
    dev = spider["validation"]
    if args.limit:
        dev = dev.select(range(min(args.limit, len(dev))))

    # Build {db_id: (tables, fks)} schema lookup. Prefer a local tables.json.
    schema_lookup = {}
    if args.tables_json:
        import json
        with open(args.tables_json, "r", encoding="utf-8") as f:
            entries = json.load(f)
    else:
        try:
            tables_ds = load_dataset("spider", "tables")
            entries = list(tables_ds[list(tables_ds.keys())[0]])
        except Exception as e:
            raise SystemExit(
                "Could not load Spider schemas. Pass --tables-json /path/to/spider/tables.json "
                "(the HF mirror has no schema config)."
            ) from e
    for entry in entries:
        schema_lookup[entry["db_id"]] = schema_from_spider_tables(entry)

    db_dir = Path(args.spider_db_dir)

    @torch.no_grad()
    def generate(question, db_id):
        tables, fks = schema_lookup.get(db_id, ({}, []))
        x = build_input(question, tables, fks)
        ids = tok(x, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        out = model.generate(ids, max_new_tokens=args.max_out, num_beams=args.num_beams)
        return tok.decode(out[0], skip_special_tokens=True)

    n = len(dev)
    exec_correct = 0
    exact_correct = 0
    missing_dbs = set()
    samples = []

    for i, ex in enumerate(dev):
        pred = generate(ex["question"], ex["db_id"])
        gold = ex["query"]

        if pred.strip().lower() == gold.strip().lower():
            exact_correct += 1

        db_path = db_dir / ex["db_id"] / f"{ex['db_id']}.sqlite"
        if db_path.exists():
            ok = exec_match(pred, gold, str(db_path))
            exec_correct += ok
        else:
            ok = None
            missing_dbs.add(ex["db_id"])

        if len(samples) < 15:
            samples.append((ex["question"], gold, pred, ok))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}  exec={exec_correct/(i+1):.3f}  exact={exact_correct/(i+1):.3f}")

    exec_acc = exec_correct / n if n else 0.0
    exact_acc = exact_correct / n if n else 0.0

    print(f"\nExecution accuracy: {exec_acc:.4f}  ({exec_correct}/{n})")
    print(f"Exact-match:        {exact_acc:.4f}  ({exact_correct}/{n})")
    if missing_dbs:
        print(f"WARNING: {len(missing_dbs)} db(s) missing under {db_dir} — those count as wrong "
              f"for execution accuracy. Missing: {sorted(missing_dbs)[:5]}...")

    write_report(args, n, exec_acc, exec_correct, exact_acc, exact_correct, samples, missing_dbs)
    print(f"\nReport written to {args.out}")


def write_report(args, n, exec_acc, exec_correct, exact_acc, exact_correct, samples, missing_dbs):
    model_id = args.model or f"{args.base_model} + adapter:{args.adapter}"
    lines = [
        "# Spider Evaluation Results",
        "",
        "> Generated by `scripts/evaluate_spider.py`. Re-run to reproduce.",
        "",
        f"- **Model:** `{model_id}`",
        f"- **Dev examples evaluated:** {n}",
        f"- **Beam size:** {args.num_beams}",
        "",
        "## Headline numbers",
        "",
        "| Metric | Score | Correct / Total |",
        "|---|---|---|",
        f"| **Execution accuracy** | **{exec_acc:.1%}** | {exec_correct} / {n} |",
        f"| Exact-match | {exact_acc:.1%} | {exact_correct} / {n} |",
        "",
    ]
    if missing_dbs:
        lines += [
            f"> ⚠️ {len(missing_dbs)} database(s) were missing under `{args.spider_db_dir}` and "
            "counted as failures. Download the full Spider DB set for an exact number.",
            "",
        ]
    lines += ["## Sample predictions", "", "| Question | Gold SQL | Predicted SQL | Exec match |",
              "|---|---|---|---|"]
    for q, gold, pred, ok in samples:
        mark = {True: "✅", False: "❌", None: "—"}[ok]
        q = q.replace("|", "\\|")
        gold = gold.replace("|", "\\|").replace("\n", " ")
        pred = pred.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {q} | `{gold}` | `{pred}` | {mark} |")
    lines.append("")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
