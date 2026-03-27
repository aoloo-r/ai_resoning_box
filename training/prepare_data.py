#!/usr/bin/env python3
"""Prepare collected ensemble data for fine-tuning.

Usage:
    python training/prepare_data.py
    python training/prepare_data.py --min-confidence 0.7 --min-models 3
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load_training_pairs(min_confidence: float = 0.6, min_models: int = 2) -> list[dict]:
    """Load and filter training pairs."""
    path = DATA_DIR / "training_pairs.jsonl"
    if not path.exists():
        print(f"No training data found at {path}")
        print("Run the ensemble platform and make queries to collect data.")
        return []

    pairs = []
    skipped = 0
    with open(path) as f:
        for line in f:
            entry = json.loads(line.strip())
            meta = entry.get("metadata", {})

            # Apply quality filters
            if meta.get("confidence", 0) < min_confidence:
                skipped += 1
                continue
            if meta.get("num_models", 0) < min_models:
                skipped += 1
                continue

            pairs.append(entry)

    print(f"Loaded {len(pairs)} training pairs ({skipped} filtered out)")
    return pairs


def to_chatml(pairs: list[dict]) -> list[dict]:
    """Convert to ChatML format for fine-tuning."""
    dataset = []
    for entry in pairs:
        dataset.append({
            "messages": [
                {"role": "system", "content": entry["system"]},
                {"role": "user", "content": entry["instruction"]},
                {"role": "assistant", "content": entry["output"]},
            ]
        })
    return dataset


def to_alpaca(pairs: list[dict]) -> list[dict]:
    """Convert to Alpaca format for fine-tuning."""
    dataset = []
    for entry in pairs:
        dataset.append({
            "instruction": entry["instruction"],
            "input": "",
            "output": entry["output"],
            "system": entry["system"],
        })
    return dataset


def to_sharegpt(pairs: list[dict]) -> list[dict]:
    """Convert to ShareGPT format for fine-tuning."""
    dataset = []
    for entry in pairs:
        dataset.append({
            "conversations": [
                {"from": "system", "value": entry["system"]},
                {"from": "human", "value": entry["instruction"]},
                {"from": "gpt", "value": entry["output"]},
            ]
        })
    return dataset


def split_dataset(dataset: list[dict], val_ratio: float = 0.1) -> tuple[list, list]:
    """Split into train/val sets."""
    random.shuffle(dataset)
    split_idx = max(1, int(len(dataset) * (1 - val_ratio)))
    return dataset[:split_idx], dataset[split_idx:]


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="Minimum confidence threshold (default: 0.6)")
    parser.add_argument("--min-models", type=int, default=2,
                        help="Minimum number of model responses (default: 2)")
    parser.add_argument("--format", choices=["chatml", "alpaca", "sharegpt"], default="chatml",
                        help="Output format (default: chatml)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    pairs = load_training_pairs(args.min_confidence, args.min_models)
    if not pairs:
        return

    # Convert to target format
    converters = {"chatml": to_chatml, "alpaca": to_alpaca, "sharegpt": to_sharegpt}
    dataset = converters[args.format](pairs)

    # Split
    train_set, val_set = split_dataset(dataset, args.val_ratio)

    # Save
    output_dir = DATA_DIR / "prepared"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    for path, data in [(train_path, train_set), (val_path, val_set)]:
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nDataset prepared ({args.format} format):")
    print(f"  Train: {len(train_set)} examples -> {train_path}")
    print(f"  Val:   {len(val_set)} examples -> {val_path}")

    # Print stats
    print(f"\nDataset statistics:")
    avg_output_len = sum(len(p["output"]) for p in pairs) / len(pairs)
    avg_input_len = sum(len(p["instruction"]) for p in pairs) / len(pairs)
    avg_conf = sum(p["metadata"]["confidence"] for p in pairs) / len(pairs)
    print(f"  Avg instruction length: {avg_input_len:.0f} chars")
    print(f"  Avg output length: {avg_output_len:.0f} chars")
    print(f"  Avg confidence: {avg_conf:.2%}")

    # Model contribution stats
    model_wins = {}
    for p in pairs:
        best = p["metadata"].get("best_model", "unknown")
        model_wins[best] = model_wins.get(best, 0) + 1
    print(f"\n  Best model distribution:")
    for model, count in sorted(model_wins.items(), key=lambda x: -x[1]):
        print(f"    {model}: {count} ({count/len(pairs):.0%})")


if __name__ == "__main__":
    main()
