#!/usr/bin/env python3
"""Fine-tune an open-source model on ensemble-collected data using LoRA.

Usage:
    # First prepare data
    python training/prepare_data.py

    # Then fine-tune
    python training/train.py
    python training/train.py --base-model Qwen/Qwen2.5-7B-Instruct --epochs 3
    python training/train.py --base-model meta-llama/Llama-3.1-8B-Instruct --lora-rank 32
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ReasoningBox model")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model from HuggingFace (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--data-dir", default=str(Path(__file__).parent / "data" / "prepared"),
                        help="Path to prepared training data")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "checkpoints"),
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank (default: 64)")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha (default: 128)")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Max sequence length (default: 4096)")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16 (default: True)")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    args = parser.parse_args()

    # Lazy imports so the script can show help without installing everything
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    if not train_path.exists():
        print("No prepared training data found!")
        print("Run: python training/prepare_data.py")
        return

    train_count = sum(1 for _ in open(train_path))
    val_count = sum(1 for _ in open(val_path)) if val_path.exists() else 0
    print(f"Training data: {train_count} examples, Validation: {val_count} examples")

    if train_count < 50:
        print(f"\nWARNING: Only {train_count} training examples. Recommended minimum is 100+.")
        print("Continue collecting data by using the ensemble platform.\n")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    data_files = {"train": str(train_path)}
    if val_path.exists() and val_count > 0:
        data_files["validation"] = str(val_path)

    dataset = load_dataset("json", data_files=data_files)

    def format_and_tokenize(examples):
        """Tokenize ChatML-format conversations."""
        texts = []
        for messages in examples["messages"]:
            # Apply chat template
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        format_and_tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    # Load model
    print(f"\nLoading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Configure LoRA
    print(f"Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_count > 0 else "no",
        save_total_limit=3,
        load_best_model_at_end=val_count > 0,
        report_to="wandb" if args.wandb else "none",
        run_name="reasoning-box-finetune",
        optim="adamw_torch",
        gradient_checkpointing=True,
        dataloader_num_workers=4,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
        data_collator=data_collator,
    )

    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max sequence length: {args.max_seq_len}")
    print()

    trainer.train()

    # Save final model
    final_dir = output_dir / "final"
    print(f"\nSaving final LoRA adapter to {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save training config for reproducibility
    config_path = final_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "base_model": args.base_model,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_seq_len": args.max_seq_len,
            "train_examples": train_count,
            "val_examples": val_count,
        }, f, indent=2)

    print("\nTraining complete!")
    print(f"  LoRA adapter saved to: {final_dir}")
    print(f"\nNext steps:")
    print(f"  1. Merge weights:  python training/export_model.py")
    print(f"  2. Test the model: python training/test_model.py")
    print(f"  3. Convert for Ollama or deploy via vLLM")


if __name__ == "__main__":
    main()
