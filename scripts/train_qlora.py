#!/usr/bin/env python3
"""
4-bit QLoRA fine-tuning of a decoder-only code LLM (default: Qwen2.5-Coder-7B-Instruct)
on the Spider text-to-SQL benchmark.

This is the *higher-accuracy* alternative to scripts/train_lora.py (which fine-tunes T5).
Decoder-only code models are much stronger at SQL, but they need a different recipe than the
seq2seq T5 path:

  * loaded in 4-bit (bitsandbytes NF4) so a 7B model fits a free ~15GB T4;
  * trained as a causal LM on prompt->completion, with the *prompt tokens masked out* of the
    loss (label = -100) so the model is only scored on the SQL it should produce;
  * the prompt is built from the shared chat template in src/schema_serialization.py, so the
    schema format never drifts from the T5 path / eval / serving.

No Unsloth and no TRL — just transformers + peft + bitsandbytes, to keep the dependency surface
(and version churn) small.

Examples
--------
# Smoke test the whole pipeline on a tiny sample:
python scripts/train_qlora.py --tables-json /path/to/spider/tables.json \
    --epochs 1 --max-train-samples 200 --output-dir smoke-qlora

# Full run on a T4 (Colab):
python scripts/train_qlora.py --tables-json /path/to/spider/tables.json \
    --epochs 2 --output-dir nl2sql-qwen-qlora

# Evaluate afterwards (note --causal):
python scripts/evaluate_spider.py --causal \
    --base-model Qwen/Qwen2.5-Coder-7B-Instruct --adapter ./nl2sql-qwen-qlora \
    --tables-json /path/to/spider/tables.json --spider-db-dir ./spider/database --limit 200

Install deps first (Colab):
    pip install "transformers>=4.44" datasets peft accelerate bitsandbytes sqlglot
"""

import argparse
import sys
from pathlib import Path

# Make `src` importable whether run from repo root or scripts/.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.schema_serialization import build_causal_messages, schema_from_spider_tables  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="4-bit QLoRA fine-tune a code LLM on Spider.")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                   help="HF decoder-only instruct model id. Smaller/faster options: "
                        "Qwen/Qwen2.5-Coder-3B-Instruct, Qwen/Qwen2.5-Coder-1.5B-Instruct.")
    p.add_argument("--output-dir", default="nl2sql-qwen-qlora",
                   help="Where to save the adapter + tokenizer.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16,
                   help="Gradient accumulation steps (effective batch = batch-size * this).")
    p.add_argument("--lr", type=float, default=2e-4, help="Typical QLoRA LR (not the 1e-3 T5 uses).")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help="Comma-separated modules to adapt (Qwen/Llama attention + MLP projections).")
    p.add_argument("--patience", type=int, default=2,
                   help="Early-stopping patience in epochs. Set 0 to disable.")
    p.add_argument("--dataloader-workers", type=int, default=2,
                   help="DataLoader worker processes (2-4 on Colab/Linux; 0 on macOS).")
    p.add_argument("--max-len", type=int, default=1024,
                   help="Max total sequence length (prompt + SQL). Long schemas may need more.")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Cap training rows (for quick smoke tests).")
    p.add_argument("--push-to-hub", default=None, help="Repo id to push the LoRA *adapter* to.")
    p.add_argument("--merge-and-push", default=None,
                   help="Repo id to push the *merged* (base+LoRA, 16-bit) model to.")
    p.add_argument("--tables-json", default=None,
                   help="Path to Spider tables.json (from the official Spider zip). Recommended.")
    p.add_argument("--dataset", default="xlangai/spider",
                   help="HF dataset id for Spider.")
    return p.parse_args()


def build_schema_lookup(tables_json=None, dataset="xlangai/spider"):
    """Build a {db_id: (tables, foreign_keys)} lookup from Spider's schema definitions."""
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
                "--tables-json /path/to/spider/tables.json (the HF mirror has no schema config)."
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
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
        Trainer, TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    if not torch.cuda.is_available():
        raise SystemExit(
            "QLoRA needs a CUDA GPU (bitsandbytes 4-bit is CUDA-only). Use scripts/train_lora.py "
            "for the T5 path on CPU/Apple Silicon, or run this on Colab/a CUDA box."
        )
    print("Compute backend: cuda")

    print(f"Loading Spider dataset ({args.dataset})...")
    spider = load_dataset(args.dataset)
    schema_lookup = build_schema_lookup(args.tables_json, args.dataset)

    print(f"Loading base model in 4-bit: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # T4 has no bf16; NF4 with fp16 compute is the right choice for a Turing GPU.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lora_targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=lora_targets, bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def preprocess(ex):
        tables, fks = schema_lookup.get(ex["db_id"], ({}, []))
        messages = build_causal_messages(ex["question"], tables, fks)
        # Build prompt (assistant turn opened, empty) and the full conversation (assistant turn
        # filled with the gold SQL) via the SAME chat template, so the special tokens / EOS are
        # exactly what the model expects — don't hand-append tok.eos_token, which may not be the
        # token the template uses to close a turn (e.g. Qwen uses <|im_end|>).
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(
            messages + [{"role": "assistant", "content": ex["query"]}], tokenize=False,
        )

        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tok(full, add_special_tokens=False)["input_ids"]

        # Mask the prompt so loss is computed only on the SQL completion.
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100

        # Truncate from the LEFT so the SQL completion (at the tail) is always preserved.
        # Right-truncating could drop the entire response, leaving labels all -100 -> NaN loss.
        if len(full_ids) > args.max_len:
            full_ids = full_ids[-args.max_len:]
            labels = labels[-args.max_len:]
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    train_split = spider["train"]
    if args.max_train_samples:
        train_split = train_split.select(range(min(args.max_train_samples, len(train_split))))

    print("Tokenizing...")
    train_ds = train_split.map(preprocess, remove_columns=train_split.column_names)
    val_ds = spider["validation"].map(preprocess, remove_columns=spider["validation"].column_names)

    # DataCollatorForSeq2Seq pads input_ids/attention_mask and pads labels with -100. Pass
    # model=None so it never tries to build encoder-decoder decoder_input_ids — we're causal.
    collator = DataCollatorForSeq2Seq(tok, model=None, label_pad_token_id=-100, padding=True)

    import inspect
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    ta_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        num_train_epochs=args.epochs,
        fp16=True,           # Qwen trains fine in fp16 (unlike T5); T4 supports fp16 tensor cores.
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # avoids requires-grad error w/ PEFT
        optim="paged_adamw_8bit",   # bitsandbytes paged optimizer — keeps optimizer state small.
        dataloader_num_workers=args.dataloader_workers,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        save_total_limit=2,
        report_to="none",
    )
    ta_kwargs["eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"] = "epoch"

    # Drop any kwargs this transformers version doesn't accept (its __init__ has no **kwargs
    # catch-all). Anything dropped here is an optimization, not correctness.
    unsupported = [k for k in ta_kwargs if k not in ta_params]
    for k in unsupported:
        ta_kwargs.pop(k)
    if unsupported:
        print(f"Note: this transformers version doesn't support {unsupported}; continuing without them.")
    training_args = TrainingArguments(**ta_kwargs)

    trainer_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = dict(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds, data_collator=collator,
    )
    trainer_kwargs["processing_class" if "processing_class" in trainer_params else "tokenizer"] = tok
    if args.patience > 0:
        trainer_kwargs["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
    trainer = Trainer(**trainer_kwargs)

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
        # You can't merge a LoRA into a 4-bit base, so reload the base in fp16 and merge there.
        # Do it on CPU: a 7B in fp16 (~14GB) won't fit alongside training state on a 15GB T4, and
        # loading on CPU avoids fighting the GPU. NOTE: still needs ~14GB *system* RAM — on a
        # free Colab (~13GB) this can OOM; if so, push the adapter instead (--push-to-hub) and
        # merge later on a bigger machine, or use a smaller --base-model.
        print("Reloading base in fp16 on CPU and merging LoRA for a 16-bit push...")
        from peft import PeftModel
        del model
        torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.float16,
            device_map="cpu", low_cpu_mem_usage=True,
        )
        merged = PeftModel.from_pretrained(base, args.output_dir).merge_and_unload()
        print(f"Pushing merged model to {args.merge_and_push}")
        merged.push_to_hub(args.merge_and_push)
        tok.push_to_hub(args.merge_and_push)

    print("Done. Next: python scripts/evaluate_spider.py --causal "
          f"--base-model {args.base_model} --adapter {args.output_dir} ...")


if __name__ == "__main__":
    main()
