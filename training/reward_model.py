#!/usr/bin/env python3
"""Reward Model — Trained on ensemble scoring data for RLHF.

The reward model learns to predict the 4-dimensional quality scores
that the ensemble judge assigns. This enables:

1. RLHF training: Use as reward signal to fine-tune ReasoningBox
2. Fast scoring: Replace expensive LLM-as-judge with learned model
3. Best-of-N sampling: Generate multiple responses, pick the best
4. Rejection sampling: Filter training data by predicted quality

The reward model is trained on (question, answer) -> score pairs
from the ensemble's scoring data.

Usage:
    python training/reward_model.py
    python training/reward_model.py --base-model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def load_reward_data(data_dir: Path, min_responses: int = 3) -> list[dict]:
    """Load and prepare reward model training data.

    Creates preference pairs from the scoring data:
    - "chosen": high-scoring response
    - "rejected": low-scoring response

    This is the standard format for reward model training.
    """
    raw_path = data_dir / "raw_ensemble.jsonl"
    if not raw_path.exists():
        print(f"No data at {raw_path}")
        return []

    pairs = []
    with open(raw_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            responses = entry.get("responses", [])

            # Need at least min_responses for meaningful comparison
            valid = [r for r in responses if not r.get("error") and r.get("content")]
            if len(valid) < min_responses:
                continue

            # Sort by score
            valid.sort(key=lambda r: r.get("total_score", 0), reverse=True)

            best = valid[0]
            worst = valid[-1]

            # Only keep if there's a meaningful score difference
            if best.get("total_score", 0) - worst.get("total_score", 0) < 0.1:
                continue

            pairs.append({
                "query": entry["query"],
                "chosen": best["content"],
                "rejected": worst["content"],
                "chosen_score": best.get("total_score", 0),
                "rejected_score": worst.get("total_score", 0),
                "chosen_scores": best.get("scores", {}),
                "rejected_scores": worst.get("scores", {}),
                "chosen_model": best.get("model_name", ""),
                "rejected_model": worst.get("model_name", ""),
            })

            # Also create pairs from adjacent rankings for more data
            for i in range(len(valid) - 1):
                better = valid[i]
                worse = valid[i + 1]
                if better.get("total_score", 0) - worse.get("total_score", 0) >= 0.05:
                    pairs.append({
                        "query": entry["query"],
                        "chosen": better["content"],
                        "rejected": worse["content"],
                        "chosen_score": better.get("total_score", 0),
                        "rejected_score": worse.get("total_score", 0),
                        "chosen_scores": better.get("scores", {}),
                        "rejected_scores": worse.get("scores", {}),
                        "chosen_model": better.get("model_name", ""),
                        "rejected_model": worse.get("model_name", ""),
                    })

    return pairs


def prepare_for_training(pairs: list[dict], tokenizer) -> list[dict]:
    """Convert to format suitable for reward model training."""
    dataset = []
    for pair in pairs:
        # Format as chat
        chosen_messages = [
            {"role": "user", "content": pair["query"]},
            {"role": "assistant", "content": pair["chosen"]},
        ]
        rejected_messages = [
            {"role": "user", "content": pair["query"]},
            {"role": "assistant", "content": pair["rejected"]},
        ]

        chosen_text = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        rejected_text = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

        dataset.append({
            "chosen": chosen_text,
            "rejected": rejected_text,
            "margin": pair["chosen_score"] - pair["rejected_score"],
        })

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train reward model")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model (smaller is fine for reward models)")
    parser.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "checkpoints" / "reward"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Load data
    print("Loading reward training data...")
    pairs = load_reward_data(data_dir)
    if not pairs:
        print("\nNo reward training data available yet.")
        print("Use the ensemble platform to collect scoring data first.")
        return

    print(f"Loaded {len(pairs)} preference pairs")

    # Lazy imports
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, TaskType
    from trl import RewardTrainer

    # Load model + tokenizer
    print(f"\nLoading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    # Prepare data
    print("Preparing training data...")
    dataset = prepare_for_training(pairs, tokenizer)

    # Save prepared data
    prepared_path = data_dir / "prepared" / "reward_train.jsonl"
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prepared_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    from datasets import load_dataset
    hf_dataset = load_dataset("json", data_files=str(prepared_path), split="train")

    def tokenize(examples):
        chosen = tokenizer(examples["chosen"], truncation=True, max_length=args.max_length, padding="max_length")
        rejected = tokenizer(examples["rejected"], truncation=True, max_length=args.max_length, padding="max_length")
        return {
            "input_ids_chosen": chosen["input_ids"],
            "attention_mask_chosen": chosen["attention_mask"],
            "input_ids_rejected": rejected["input_ids"],
            "attention_mask_rejected": rejected["attention_mask"],
        }

    tokenized = hf_dataset.map(tokenize, batched=True, remove_columns=hf_dataset.column_names)

    # Training
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        peft_config=lora_config,
    )

    print(f"\nTraining reward model...")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Epochs: {args.epochs}")
    trainer.train()

    # Save
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save metadata
    with open(final_dir / "reward_config.json", "w") as f:
        json.dump({
            "base_model": args.base_model,
            "training_pairs": len(pairs),
            "lora_rank": args.lora_rank,
            "epochs": args.epochs,
        }, f, indent=2)

    print(f"\nReward model saved to {final_dir}")
    print("\nNext steps:")
    print("  1. Use for RLHF: Integrate with training/train.py using PPO/DPO")
    print("  2. Use for scoring: Replace LLM-as-judge for fast evaluation")
    print("  3. Use for filtering: Score and filter training data quality")


if __name__ == "__main__":
    main()
