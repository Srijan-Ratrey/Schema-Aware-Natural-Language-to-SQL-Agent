#!/usr/bin/env python3
"""
LoRA fine-tuning of T5 on the Spider text-to-SQL benchmark.

This is the *real* fine-tuning entry point referenced by docs/NL2SQL_WALKTHROUGH.md
and docs/GAP_ANALYSIS.md. It is designed to run on a single free GPU (Colab/Kaggle T4) or locally on Apple Silicon
(MPS): LoRA on the attention q/k/v/o projections, fp32/bf16 (never fp16 — T5 NaNs),
gradient checkpointing, small batch + accumulation, a cosine LR schedule with warmup,
early stopping, and best-checkpoint selection.

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
    p.add_argument("--lora-targets", default="q,k,v,o",
                   help="Comma-separated T5 modules to adapt. Default 'q,k,v,o' (wider than the "
                        "old 'q,v') lifts accuracy; add 'wi,wo' for the feed-forward layers at "
                        "higher memory/time cost.")
    p.add_argument("--patience", type=int, default=2,
                   help="Early-stopping patience in epochs (stop if eval loss doesn't improve). "
                        "Set 0 to disable early stopping.")
    p.add_argument("--dataloader-workers", type=int, default=0,
                   help="DataLoader worker processes. Keep 0 on macOS/MPS to avoid fork issues; "
                        "2-4 on Colab/Linux keeps the GPU fed.")
    p.add_argument("--no-grad-checkpointing", action="store_true",
                   help="Disable gradient checkpointing. Big speedup when the model leaves VRAM "
                        "free (e.g. t5-base on a 15GB T4); leave it on if you hit OOM.")
    p.add_argument("--max-in", type=int, default=512)
    p.add_argument("--max-out", type=int, default=256)
    p.add_argument("--no-fp16", action="store_true",
                   help="Force fp32 (disable bf16). T5 is never run in fp16 — it NaNs.")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Cap training rows (for quick smoke tests).")
    p.add_argument("--push-to-hub", default=None,
                   help="Repo id to push the LoRA *adapter* to.")
    p.add_argument("--merge-and-push", default=None,
                   help="Repo id to push the *merged* (base+LoRA) model to.")
    p.add_argument("--tables-json", default=None,
                   help="Path to Spider tables.json (from the official Spider zip). "
                        "Recommended — the HF mirror does not ship schema info.")
    p.add_argument("--dataset", default="xlangai/spider",
                   help="HF dataset id for Spider (bare 'spider' is no longer loadable on "
                        "recent datasets/huggingface_hub).")
    return p.parse_args()


def build_schema_lookup(tables_json=None, dataset="xlangai/spider"):
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
            tables_ds = load_dataset(dataset, "tables")
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

    # On Apple Silicon (MPS) a handful of T5 ops have no Metal kernel; let them fall back
    # to CPU instead of raising. Harmless on CUDA/CPU. Set before importing torch.
    import os
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    # Report the compute backend the Trainer will use so `mps`/`cuda`/`cpu` is obvious in logs.
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Compute backend: {device}")

    print(f"Loading Spider dataset ({args.dataset})...")
    spider = load_dataset(args.dataset)
    schema_lookup = build_schema_lookup(args.tables_json, args.dataset)

    print(f"Loading base model: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    # Gradient checkpointing recomputes activations on the backward pass to save memory, at a
    # real speed cost. It's on by default so big models fit a small GPU, but for a model that
    # leaves lots of VRAM free (e.g. t5-base on a 15GB T4) turning it off is a large speedup.
    if not args.no_grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # required with gradient checkpointing

    lora_targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=lora_targets,  # e.g. q/k/v/o attention projections (+ wi/wo FF)
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

    # Build args/trainer in a version-tolerant way: the HF API renamed a couple of
    # kwargs across versions (evaluation_strategy -> eval_strategy, and the Trainer's
    # tokenizer -> processing_class in transformers 4.46+). Detect what's supported so
    # this runs on both older pinned and the latest Colab transformers.
    import inspect
    ta_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters

    # CRITICAL: the original T5 checkpoints (t5-small/base/large) were trained in bf16 and
    # are numerically unstable in fp16 — fp16 produces NaN losses, training collapses, and
    # the model degenerates to emitting a single high-frequency token (e.g. ","). So NEVER
    # use fp16 for T5. Prefer bf16 where the GPU supports it, otherwise fall back to fp32.
    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and not args.no_fp16
    )
    ta_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        bf16=use_bf16,
        fp16=False,
        # No compute_metrics is attached, so generation during eval would be pure wasted compute
        # (its output is never scored). Keep it off — eval reports loss only, which is what we use
        # for best-checkpoint selection. Real execution accuracy is measured by evaluate_spider.py.
        predict_with_generate=False,
        group_by_length=True,           # batch similar-length sequences -> less padding waste
        dataloader_num_workers=args.dataloader_workers,
        save_strategy="epoch",
        load_best_model_at_end=True,    # keep the best-generalizing adapter, not just the last epoch
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )
    ta_kwargs["eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"] = "epoch"

    # Drop any kwargs this transformers version doesn't accept (its __init__ has no **kwargs
    # catch-all, so an unknown key raises). transformers 5.x removed a few args (e.g.
    # group_by_length). The ones dropped here are pure optimizations, not correctness.
    unsupported = [k for k in ta_kwargs if k not in ta_params]
    for k in unsupported:
        ta_kwargs.pop(k)
    if unsupported:
        print(f"Note: this transformers version doesn't support {unsupported}; continuing without them.")

    training_args = Seq2SeqTrainingArguments(**ta_kwargs)
    print(f"Precision: {'bf16' if use_bf16 else 'fp32'} (fp16 disabled for T5 stability)")

    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    trainer_kwargs = dict(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer_kwargs["processing_class" if "processing_class" in trainer_params else "tokenizer"] = tok
    if args.patience > 0:
        trainer_kwargs["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
    trainer = Seq2SeqTrainer(**trainer_kwargs)

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
