"""JEPA World Model — Predicts reasoning quality and guides generation.

The World Model uses the trained JEPA representations to:
1. Predict how good a response will be BEFORE generating it
2. Score responses in embedding space (faster than LLM-as-judge)
3. Guide decoding by steering toward high-quality reasoning regions
4. Detect reasoning failures and hallucinations

This is the key differentiator: instead of just generating text,
we predict the abstract quality of reasoning and use that to
produce better outputs.
"""

from __future__ import annotations
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class WorldModel(nn.Module):
    """World model that predicts reasoning outcomes in embedding space.

    Components:
        1. Quality Predictor: Given question embedding, predicts
           quality scores (accuracy, completeness, reasoning, clarity)
        2. Difficulty Estimator: Predicts how hard a question is
           (determines how many reasoning steps to use)
        3. Strategy Selector: Predicts which synthesis strategy
           will work best for this question
        4. Reasoning Planner: Plans the reasoning steps needed
    """

    def __init__(
        self,
        d_embedding: int = 512,
        d_hidden: int = 1024,
        n_strategies: int = 4,
        n_difficulty_levels: int = 5,
        n_reasoning_steps: int = 8,
    ):
        super().__init__()
        self.d_embedding = d_embedding

        # Quality predictor: embedding -> 4 quality scores
        self.quality_head = nn.Sequential(
            nn.Linear(d_embedding, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 4),  # accuracy, completeness, reasoning, clarity
            nn.Sigmoid(),
        )

        # Difficulty estimator
        self.difficulty_head = nn.Sequential(
            nn.Linear(d_embedding, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, n_difficulty_levels),
        )

        # Strategy selector
        self.strategy_head = nn.Sequential(
            nn.Linear(d_embedding, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, n_strategies),
        )

        # Reasoning planner: predicts sequence of reasoning aspects needed
        self.plan_head = nn.Sequential(
            nn.Linear(d_embedding, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_reasoning_steps * d_embedding),
        )
        self.n_reasoning_steps = n_reasoning_steps

        # Confidence calibrator: learns to estimate prediction uncertainty
        self.confidence_head = nn.Sequential(
            nn.Linear(d_embedding * 2, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Sigmoid(),
        )

        # Hallucination detector: predicts if an answer is grounded
        self.hallucination_head = nn.Sequential(
            nn.Linear(d_embedding * 2, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Sigmoid(),
        )

    def predict_quality(self, z_question: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict expected quality scores for a question.

        Given just the question embedding (no answer yet), predict
        what quality scores a good answer should achieve.

        Returns dict with:
            scores: (B, 4) predicted [accuracy, completeness, reasoning, clarity]
            difficulty: (B,) difficulty level 0-4
            strategy: (B,) recommended strategy index
            plan: (B, n_steps, d_embedding) reasoning step embeddings
        """
        scores = self.quality_head(z_question)
        difficulty_logits = self.difficulty_head(z_question)
        strategy_logits = self.strategy_head(z_question)

        plan = self.plan_head(z_question)
        plan = plan.view(-1, self.n_reasoning_steps, self.d_embedding)

        return {
            "scores": scores,
            "score_labels": ["accuracy", "completeness", "reasoning_quality", "clarity"],
            "difficulty": difficulty_logits.argmax(dim=-1),
            "difficulty_logits": difficulty_logits,
            "strategy": strategy_logits.argmax(dim=-1),
            "strategy_logits": strategy_logits,
            "reasoning_plan": plan,
        }

    def evaluate_answer(
        self,
        z_question: torch.Tensor,
        z_answer: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate an answer's quality using embeddings (fast, no LLM needed).

        This replaces the expensive LLM-as-judge scoring with a learned
        embedding-space evaluation.
        """
        combined = torch.cat([z_question, z_answer], dim=-1)

        confidence = self.confidence_head(combined).squeeze(-1)
        hallucination_prob = self.hallucination_head(combined).squeeze(-1)
        quality = self.quality_head(z_answer)

        # Overall score: weighted combination
        weights = torch.tensor([0.35, 0.25, 0.20, 0.10], device=quality.device)
        overall = (quality * weights).sum(dim=-1)

        return {
            "overall_score": overall,
            "quality_scores": quality,
            "confidence": confidence,
            "hallucination_probability": hallucination_prob,
            "is_grounded": hallucination_prob < 0.3,
        }

    def compute_loss(
        self,
        z_question: torch.Tensor,
        z_answer: torch.Tensor,
        true_scores: torch.Tensor,
        true_difficulty: torch.Tensor | None = None,
        true_strategy: torch.Tensor | None = None,
        is_hallucination: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute training loss for the world model.

        Args:
            z_question: (B, D) question embeddings
            z_answer: (B, D) answer embeddings
            true_scores: (B, 4) ground truth quality scores
            true_difficulty: (B,) ground truth difficulty levels
            true_strategy: (B,) ground truth best strategy indices
            is_hallucination: (B,) 1.0 if hallucinated, 0.0 if grounded
        """
        losses = {}

        # Quality prediction loss
        pred_quality = self.quality_head(z_answer)
        losses["quality"] = F.mse_loss(pred_quality, true_scores)

        # Difficulty prediction loss
        if true_difficulty is not None:
            diff_logits = self.difficulty_head(z_question)
            losses["difficulty"] = F.cross_entropy(diff_logits, true_difficulty)

        # Strategy prediction loss
        if true_strategy is not None:
            strat_logits = self.strategy_head(z_question)
            losses["strategy"] = F.cross_entropy(strat_logits, true_strategy)

        # Hallucination detection loss
        if is_hallucination is not None:
            combined = torch.cat([z_question, z_answer], dim=-1)
            hall_pred = self.hallucination_head(combined).squeeze(-1)
            losses["hallucination"] = F.binary_cross_entropy(hall_pred, is_hallucination)

        # Total
        losses["total"] = sum(losses.values())
        return losses


class GuidedDecoder:
    """Uses the JEPA World Model to guide text generation.

    During generation, the world model:
    1. Predicts what good reasoning looks like (target embedding)
    2. At each step, scores candidate continuations
    3. Steers generation toward the predicted reasoning space
    4. Detects and avoids hallucination regions

    This is like having a "reasoning compass" that points the
    generator toward the best possible answer.
    """

    def __init__(
        self,
        jepa_model,
        world_model: WorldModel,
        tokenizer,
        guidance_strength: float = 2.0,
        hallucination_penalty: float = 5.0,
    ):
        self.jepa = jepa_model
        self.world = world_model
        self.tokenizer = tokenizer
        self.guidance_strength = guidance_strength
        self.hallucination_penalty = hallucination_penalty

    @torch.no_grad()
    def plan_reasoning(self, question: str) -> dict:
        """Create a reasoning plan before generating.

        Returns a plan with:
        - Expected quality targets
        - Difficulty level
        - Recommended strategy
        - Reasoning step embeddings (the "compass points")
        """
        tokens = self.tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
        input_ids = tokens["input_ids"].to(next(self.jepa.parameters()).device)
        mask = tokens.get("attention_mask", None)
        if mask is not None:
            mask = mask.to(input_ids.device)

        # Get question embedding
        z_question = self.jepa.encode_question(input_ids, mask)

        # Get world model predictions
        predictions = self.world.predict_quality(z_question)

        strategy_names = ["weighted_merge", "best_of_n", "debate", "chain_of_verification"]
        difficulty_names = ["trivial", "easy", "medium", "hard", "expert"]

        return {
            "question_embedding": z_question,
            "target_quality": {
                "accuracy": predictions["scores"][0, 0].item(),
                "completeness": predictions["scores"][0, 1].item(),
                "reasoning_quality": predictions["scores"][0, 2].item(),
                "clarity": predictions["scores"][0, 3].item(),
            },
            "difficulty": difficulty_names[predictions["difficulty"][0].item()],
            "recommended_strategy": strategy_names[predictions["strategy"][0].item()],
            "reasoning_plan": predictions["reasoning_plan"],
            "num_reasoning_steps": max(2, predictions["difficulty"][0].item() + 2),
        }

    @torch.no_grad()
    def score_response(self, question: str, answer: str) -> dict:
        """Score a response using the world model (fast, no LLM needed)."""
        device = next(self.jepa.parameters()).device

        q_tokens = self.tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
        a_tokens = self.tokenizer(answer, return_tensors="pt", truncation=True, max_length=2048)

        q_ids = q_tokens["input_ids"].to(device)
        q_mask = q_tokens.get("attention_mask", torch.ones_like(q_ids)).to(device)
        a_ids = a_tokens["input_ids"].to(device)
        a_mask = a_tokens.get("attention_mask", torch.ones_like(a_ids)).to(device)

        z_question = self.jepa.encode_question(q_ids, q_mask)
        z_answer = self.jepa.encode_answer(a_ids, a_mask)

        evaluation = self.world.evaluate_answer(z_question, z_answer)

        return {
            "overall_score": evaluation["overall_score"][0].item(),
            "quality": {
                "accuracy": evaluation["quality_scores"][0, 0].item(),
                "completeness": evaluation["quality_scores"][0, 1].item(),
                "reasoning_quality": evaluation["quality_scores"][0, 2].item(),
                "clarity": evaluation["quality_scores"][0, 3].item(),
            },
            "confidence": evaluation["confidence"][0].item(),
            "hallucination_probability": evaluation["hallucination_probability"][0].item(),
            "is_grounded": evaluation["is_grounded"][0].item(),
            "reasoning_similarity": self.jepa.similarity(z_question, z_answer)[0].item(),
        }

    @torch.no_grad()
    def rerank_responses(self, question: str, answers: list[str]) -> list[dict]:
        """Rerank multiple answers by reasoning quality (fast batch scoring)."""
        scores = [self.score_response(question, a) for a in answers]
        ranked = sorted(enumerate(scores), key=lambda x: -x[1]["overall_score"])
        return [
            {"rank": i + 1, "index": idx, "answer": answers[idx], **score}
            for i, (idx, score) in enumerate(ranked)
        ]
