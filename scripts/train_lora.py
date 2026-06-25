#!/usr/bin/env python3
"""
LoRA fine-tuning of T5 on the Spider text-to-SQL benchmark.

This is the *real* fine-tuning entry point referenced by docs/NL2SQL_WALKTHROUGH.md
and docs/GAP_ANALYSIS.md. It is designed to run on a single free GPU (Colab/Kaggle T4):
LoRA on the attention q/v projections, fp16, gradient checkpointing, small batch +
accumulation, and a checkpoint every epoch.

Schema serialization is delegated to `src/schema_serialization.py` so the exact format
used here is the same one used at eval and serving time.

Examples
--------
# Quick smoke test of the whole pipeline on the small base model:
python scripts/train_lora.py --base-model t5-base --epochs 1 --max-train-samples 200

# Full run, strongest accuracy (needs ~12GB VRAM):
python scripts/train_lora.py --base-model t5-large --epochs 5

# Train, merge the adapter into the base, and push the merged model to the Hub:
python scripts/train_lora.py --base-model t5-large --epochs 5 \
    --merge-and-push your-username/nl2sql-t5-large-spider

Install deps first:
    pip install transformers datasets peft accelerate evaluate sqlglot sentencepiece
"""

import argparse
import sys
from pathlib import Path

# Make `src` importable whether run from repo root or scripts/.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.schema_serialization import build_input, schema_from_spider_tables  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tune T5 on Spider.")
    p.add_argument("--base-model", default="t5-large",
                   help="HF base model id (t5-small | t5-base | t5-large).")
    p.add_argument("--output-dir", default="nl2sql-t5-lora",
                   help="Where to save the adapter + tokenizer.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch-size * this).")
    p.add_argument("--lr", type=float, default=1e-3, help="LoRA tolerates a high LR.")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-in", type=int, default=512)
    p.add_argument("--max-out", type=int, default=256)
    p.add_argument("--no-fp16", action="store_true", help="Disable fp16 (use on CPU/MPS).")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Cap training rows (for quick smoke tests).")
    p.add_argument("--push-to-hub", default=None,
                   help="Repo id to push the LoRA *adapter* to.")
    p.add_argument("--merge-and-push", default=None,
                   help="Repo id to push the *merged* (base+LoRA) model to.")
    p.add_argument("--tables-json", default=None,
                   help="Path to Spider tables.json (from the official Spider zip). "
                        "Recommended — the HF mirror does not ship schema info.")
    return p.parse_args()


def build_schema_lookup(tables_json=None):
    """
    Build a {db_id: (tables, foreign_keys)} lookup from Spider's schema definitions.

    Preferred source is a local tables.json from the official Spider download
    (pass --tables-json). As a fallback we try the HF 'tables' config, which most
    mirrors do NOT provide.
    """
    import json

    if tables_json:
        with open(tables_json, "r", encoding="utf-8") as f:
            entries = json.load(f)
    else:
        from datasets import load_dataset
        try:
            tables_ds = load_dataset("spider", "tables")
            entries = list(tables_ds[list(tables_ds.keys())[0]])
        except Exception as e:
            raise RuntimeError(
                "Could not load Spider schemas. Download the official Spider zip and pass "
                "--tables-json /path/to/spider/tables.json (the HF mirror has no schema "
                "config). See docs/NL2SQL_WALKTHROUGH.md."
            ) from e

    lookup = {}
    for entry in entries:
        tables, fks = schema_from_spider_tables(entry)
        lookup[entry["db_id"]] = (tables, fks)
    return lookup


def main():
    args = parse_args()

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"Loading Spider dataset...")
    spider = load_dataset("spider")
    schema_lookup = build_schema_lookup(args.tables_json)

    print(f"Loading base model: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # required with gradient checkpointing

    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q", "v"],  # T5 attention query/value projections
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def preprocess(ex):
        tables, fks = schema_lookup.get(ex["db_id"], ({}, []))
        x = build_input(ex["question"], tables, fks)
        model_in = tok(x, max_length=args.max_in, truncation=True)
        labels = tok(text_target=ex["query"], max_length=args.max_out, truncation=True)
        model_in["labels"] = labels["input_ids"]
        return model_in

    train_split = spider["train"]
    if args.max_train_samples:
        train_split = train_split.select(range(min(args.max_train_samples, len(train_split))))

    print("Tokenizing...")
    train_ds = train_split.map(preprocess, remove_columns=train_split.column_names)
    val_ds = spider["validation"].map(preprocess, remove_columns=spider["validation"].column_names)

    collator = DataCollatorForSeq2Seq(tok, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=(not args.no_fp16 and torch.cuda.is_available()),
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, tokenizer=tok,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter + tokenizer to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.push_to_hub:
        print(f"Pushing LoRA adapter to {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub)
        tok.push_to_hub(args.push_to_hub)

    if args.merge_and_push:
        print("Merging LoRA into base model...")
        merged = model.merge_and_unload()
        print(f"Pushing merged model to {args.merge_and_push}")
        merged.push_to_hub(args.merge_and_push)
        tok.push_to_hub(args.merge_and_push)

    print("Done. Next: python scripts/evaluate_spider.py "
          f"--model {args.merge_and_push or args.output_dir}")


if __name__ == "__main__":
    main()
