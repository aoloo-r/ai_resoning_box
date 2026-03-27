#!/usr/bin/env python3
"""Train the JEPA Reasoning Model.

This trains the Joint Embedding Predictive Architecture to learn
abstract reasoning patterns in embedding space.

Training data: (question, good_answer, bad_answer) triplets from
the ensemble's scored responses.

Loss: VICReg (Variance-Invariance-Covariance) + Contrastive margin
- Invariance: Pull predicted embedding toward good answer embedding
- Variance: Prevent representation collapse
- Covariance: Decorrelate embedding dimensions
- Contrastive: Push prediction away from bad answer embeddings

Usage:
    python training/train_jepa.py
    python training/train_jepa.py --d-model 768 --epochs 50 --batch-size 32
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ReasoningTripletDataset(Dataset):
    """Dataset of (question, good_answer, bad_answer) triplets."""

    def __init__(self, data_path: Path, tokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.triplets = []

        if not data_path.exists():
            return

        with open(data_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                responses = entry.get("responses", [])
                valid = [r for r in responses if not r.get("error") and r.get("content")]

                if len(valid) < 2:
                    continue

                valid.sort(key=lambda r: r.get("total_score", 0), reverse=True)

                # Create triplets: question + best answer + each worse answer
                best = valid[0]
                for worse in valid[1:]:
                    if best.get("total_score", 0) - worse.get("total_score", 0) >= 0.05:
                        self.triplets.append({
                            "question": entry["query"],
                            "good_answer": best["content"],
                            "bad_answer": worse["content"],
                            "good_score": best.get("total_score", 0),
                            "bad_score": worse.get("total_score", 0),
                        })

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx]

        q = self.tokenizer(
            t["question"], truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )
        good = self.tokenizer(
            t["good_answer"], truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )
        bad = self.tokenizer(
            t["bad_answer"], truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )

        return {
            "question_ids": q["input_ids"].squeeze(0),
            "question_mask": q["attention_mask"].squeeze(0),
            "good_ids": good["input_ids"].squeeze(0),
            "good_mask": good["attention_mask"].squeeze(0),
            "bad_ids": bad["input_ids"].squeeze(0),
            "bad_mask": bad["attention_mask"].squeeze(0),
            "good_score": torch.tensor(t["good_score"], dtype=torch.float32),
            "bad_score": torch.tensor(t["bad_score"], dtype=torch.float32),
        }


def main():
    parser = argparse.ArgumentParser(description="Train JEPA Reasoning Model")
    parser.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "checkpoints" / "jepa"))
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Tokenizer to use for text encoding")
    parser.add_argument("--d-model", type=int, default=768, help="Transformer hidden dim")
    parser.add_argument("--d-projection", type=int, default=384, help="Projection/embedding dim")
    parser.add_argument("--encoder-layers", type=int, default=8, help="Encoder transformer layers")
    parser.add_argument("--predictor-layers", type=int, default=4, help="Predictor layers")
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from core.jepa.architecture import ReasoningJEPA
    from core.jepa.world_model import WorldModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # Load data
    data_path = Path(args.data_dir) / "raw_ensemble.jsonl"
    print(f"Loading training data from {data_path}")
    dataset = ReasoningTripletDataset(data_path, tokenizer, max_len=args.max_seq_len)

    if len(dataset) == 0:
        print("\nNo training triplets available.")
        print("Use the ensemble platform to collect data first.")
        print("Need at least a few queries with 2+ model responses.")
        return

    print(f"Training triplets: {len(dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    # Create JEPA model
    print(f"\nCreating JEPA model (d={args.d_model}, proj={args.d_projection})")
    jepa = ReasoningJEPA(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_projection=args.d_projection,
        encoder_layers=args.encoder_layers,
        predictor_layers=args.predictor_layers,
        n_heads=args.n_heads,
        ema_decay=args.ema_decay,
    ).to(device)

    # Create World Model
    world_model = WorldModel(d_embedding=args.d_projection).to(device)

    # Count parameters
    jepa_params = sum(p.numel() for p in jepa.parameters())
    wm_params = sum(p.numel() for p in world_model.parameters())
    print(f"JEPA parameters: {jepa_params:,}")
    print(f"World Model parameters: {wm_params:,}")
    print(f"Total: {jepa_params + wm_params:,}")

    # Optimizers
    jepa_optimizer = optim.AdamW(
        list(jepa.context_encoder.parameters()) + list(jepa.predictor.parameters()),
        lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95),
    )
    wm_optimizer = optim.AdamW(
        world_model.parameters(),
        lr=args.lr * 0.5, weight_decay=0.01,
    )

    # LR scheduler with warmup
    def lr_schedule(epoch):
        if epoch < args.warmup_epochs:
            return epoch / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    jepa_scheduler = optim.lr_scheduler.LambdaLR(jepa_optimizer, lr_schedule)
    wm_scheduler = optim.lr_scheduler.LambdaLR(wm_optimizer, lr_schedule)

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting JEPA training for {args.epochs} epochs...")
    print(f"{'Epoch':>6} {'Loss':>10} {'Invar':>8} {'Var':>8} {'Cov':>8} {'Contr':>8} {'WM':>8} {'LR':>10}")
    print("-" * 80)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        jepa.train()
        world_model.train()

        epoch_losses = {"total": 0, "invariance": 0, "variance": 0, "covariance": 0, "contrastive": 0, "world": 0}
        num_batches = 0

        for batch in dataloader:
            # Move to device
            q_ids = batch["question_ids"].to(device)
            q_mask = batch["question_mask"].to(device)
            g_ids = batch["good_ids"].to(device)
            g_mask = batch["good_mask"].to(device)
            b_ids = batch["bad_ids"].to(device)
            b_mask = batch["bad_mask"].to(device)
            good_scores = batch["good_score"].to(device)
            bad_scores = batch["bad_score"].to(device)

            # === JEPA Forward ===
            jepa_losses = jepa(
                question_ids=q_ids, question_mask=q_mask,
                answer_ids=g_ids, answer_mask=g_mask,
                negative_ids=b_ids, negative_mask=b_mask,
            )

            jepa_optimizer.zero_grad()
            jepa_losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(jepa.parameters(), 1.0)
            jepa_optimizer.step()

            # Update target encoder (EMA)
            jepa.update_target_encoder()

            # === World Model Forward ===
            with torch.no_grad():
                z_question = jepa.context_encoder(q_ids, q_mask)
                z_good = jepa.target_encoder(g_ids, g_mask)

            # Create pseudo quality scores from the ensemble scores
            # Normalize to 4 dimensions
            quality_targets = torch.stack([
                good_scores, good_scores * 0.9,
                good_scores * 0.85, good_scores * 0.95
            ], dim=-1)

            wm_losses = world_model.compute_loss(z_question, z_good, quality_targets)

            wm_optimizer.zero_grad()
            wm_losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
            wm_optimizer.step()

            # Track losses
            epoch_losses["total"] += jepa_losses["total"].item() + wm_losses["total"].item()
            epoch_losses["invariance"] += jepa_losses["invariance"].item()
            epoch_losses["variance"] += jepa_losses["variance"].item()
            epoch_losses["covariance"] += jepa_losses["covariance"].item()
            epoch_losses["contrastive"] += jepa_losses.get("contrastive", torch.tensor(0)).item()
            epoch_losses["world"] += wm_losses["total"].item()
            num_batches += 1

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)

        jepa_scheduler.step()
        wm_scheduler.step()

        lr = jepa_optimizer.param_groups[0]["lr"]
        print(
            f"{epoch+1:>6} {epoch_losses['total']:>10.4f} "
            f"{epoch_losses['invariance']:>8.4f} {epoch_losses['variance']:>8.4f} "
            f"{epoch_losses['covariance']:>8.4f} {epoch_losses['contrastive']:>8.4f} "
            f"{epoch_losses['world']:>8.4f} {lr:>10.6f}"
        )

        # Save best
        if epoch_losses["total"] < best_loss:
            best_loss = epoch_losses["total"]
            torch.save({
                "epoch": epoch,
                "jepa_state": jepa.state_dict(),
                "world_model_state": world_model.state_dict(),
                "loss": best_loss,
                "config": {
                    "vocab_size": vocab_size,
                    "d_model": args.d_model,
                    "d_projection": args.d_projection,
                    "encoder_layers": args.encoder_layers,
                    "predictor_layers": args.predictor_layers,
                    "n_heads": args.n_heads,
                    "tokenizer": args.tokenizer,
                },
            }, output_dir / "best.pt")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "jepa_state": jepa.state_dict(),
                "world_model_state": world_model.state_dict(),
                "jepa_optimizer": jepa_optimizer.state_dict(),
                "wm_optimizer": wm_optimizer.state_dict(),
                "loss": epoch_losses["total"],
            }, output_dir / f"checkpoint_{epoch+1}.pt")

    # Save final
    torch.save({
        "epoch": args.epochs,
        "jepa_state": jepa.state_dict(),
        "world_model_state": world_model.state_dict(),
        "loss": epoch_losses["total"],
        "config": {
            "vocab_size": vocab_size,
            "d_model": args.d_model,
            "d_projection": args.d_projection,
            "encoder_layers": args.encoder_layers,
            "predictor_layers": args.predictor_layers,
            "n_heads": args.n_heads,
            "tokenizer": args.tokenizer,
        },
    }, output_dir / "final.pt")

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Models saved to {output_dir}")
    print(f"\nThe JEPA model can now:")
    print(f"  1. Predict reasoning quality before generation")
    print(f"  2. Score responses in embedding space (fast)")
    print(f"  3. Guide decoding toward high-quality reasoning")
    print(f"  4. Detect hallucinations and weak reasoning")


if __name__ == "__main__":
    main()
