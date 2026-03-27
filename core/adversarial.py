"""Adversarial Weakness Finder — Discovers where frontier models fail.

Analyzes collected ensemble data to find:
1. Question types where models consistently score low
2. Topics where models disagree (uncertain knowledge)
3. Reasoning patterns that trip up specific models
4. Edge cases that expose model blindspots

Then generates targeted training data to make ReasoningBox
strong exactly where Claude, GPT-4, and Gemini are weak.

The key insight: you don't need to be better than frontier models
at EVERYTHING — you need to be better where they FAIL.
"""

from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class WeaknessPattern:
    """A discovered weakness pattern in frontier models."""
    category: str  # e.g., "logical_reasoning", "math", "ambiguous_queries"
    description: str
    severity: float  # 0-1, how badly models fail
    affected_models: list[str]
    example_queries: list[str]
    avg_score: float
    disagreement_rate: float
    count: int


@dataclass
class AdversarialExample:
    """A training example specifically targeting a model weakness."""
    query: str
    best_answer: str
    weakness_category: str
    target_score_improvement: float
    failed_model_answers: list[dict]  # What the models got wrong


class WeaknessFinder:
    """Analyzes ensemble data to find frontier model weaknesses."""

    # Question category patterns
    CATEGORIES = {
        "logical_reasoning": [
            r"if.*then", r"logic", r"paradox", r"fallacy", r"syllogism",
            r"deduc", r"induc", r"implies", r"contradict",
        ],
        "mathematical": [
            r"calculat", r"equation", r"prove", r"theorem", r"integral",
            r"probability", r"statistic", r"algebra", r"geometric",
        ],
        "coding": [
            r"code", r"program", r"function", r"algorithm", r"debug",
            r"implement", r"python", r"javascript", r"sql", r"api",
        ],
        "factual_recall": [
            r"when did", r"who was", r"what is the", r"how many",
            r"capital of", r"founded", r"invented", r"discovered",
        ],
        "creative": [
            r"write a", r"story", r"poem", r"creative", r"imagine",
            r"design", r"brainstorm", r"invent",
        ],
        "ethical_nuanced": [
            r"should", r"ethical", r"moral", r"dilemma", r"fair",
            r"bias", r"controversy", r"opinion",
        ],
        "multi_step": [
            r"step by step", r"first.*then", r"plan", r"strategy",
            r"compare.*and", r"pros.*cons", r"trade.?off",
        ],
        "ambiguous": [
            r"what do you think", r"it depends", r"could be",
            r"interpret", r"meaning of", r"define",
        ],
    }

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent / "training" / "data"
        self.raw_file = self.data_dir / "raw_ensemble.jsonl"

    def load_data(self) -> list[dict]:
        """Load all raw ensemble results."""
        if not self.raw_file.exists():
            return []
        entries = []
        with open(self.raw_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def categorize_query(self, query: str) -> list[str]:
        """Categorize a query into weakness categories."""
        query_lower = query.lower()
        categories = []
        for cat, patterns in self.CATEGORIES.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    categories.append(cat)
                    break
        return categories or ["general"]

    def analyze_weaknesses(self) -> list[WeaknessPattern]:
        """Analyze all collected data to find weakness patterns."""
        data = self.load_data()
        if not data:
            return []

        # Group by category
        category_stats = defaultdict(lambda: {
            "queries": [],
            "scores": [],
            "model_scores": defaultdict(list),
            "disagreements": 0,
            "total": 0,
            "low_confidence": 0,
        })

        for entry in data:
            categories = self.categorize_query(entry["query"])
            for cat in categories:
                stats = category_stats[cat]
                stats["queries"].append(entry["query"])
                stats["total"] += 1

                if entry.get("disagreement_points"):
                    stats["disagreements"] += 1
                if entry.get("confidence", 1.0) < 0.6:
                    stats["low_confidence"] += 1

                for resp in entry.get("responses", []):
                    score = resp.get("total_score", 0)
                    model = resp.get("model_name", "unknown")
                    stats["scores"].append(score)
                    stats["model_scores"][model].append(score)

        # Find weakness patterns
        weaknesses = []
        for cat, stats in category_stats.items():
            if stats["total"] < 2:
                continue

            avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
            disagreement_rate = stats["disagreements"] / stats["total"]

            # Find which models struggle most in this category
            model_avgs = {}
            for model, scores in stats["model_scores"].items():
                model_avgs[model] = sum(scores) / len(scores)

            weak_models = [m for m, s in model_avgs.items() if s < avg_score * 0.85]

            # Severity: combination of low scores, high disagreement, and low confidence
            severity = (
                (1.0 - avg_score) * 0.4 +
                disagreement_rate * 0.3 +
                (stats["low_confidence"] / stats["total"]) * 0.3
            )

            if severity > 0.2 or avg_score < 0.7:
                weaknesses.append(WeaknessPattern(
                    category=cat,
                    description=self._describe_weakness(cat, avg_score, weak_models, disagreement_rate),
                    severity=min(1.0, severity),
                    affected_models=weak_models or list(model_avgs.keys()),
                    example_queries=stats["queries"][:5],
                    avg_score=avg_score,
                    disagreement_rate=disagreement_rate,
                    count=stats["total"],
                ))

        return sorted(weaknesses, key=lambda w: -w.severity)

    def _describe_weakness(
        self, category: str, avg_score: float, weak_models: list[str], disagreement_rate: float
    ) -> str:
        """Generate a human-readable description of a weakness."""
        parts = [f"Models struggle with {category.replace('_', ' ')} questions"]
        parts.append(f"(avg score: {avg_score:.0%})")
        if weak_models:
            parts.append(f"Weakest: {', '.join(weak_models[:3])}")
        if disagreement_rate > 0.3:
            parts.append(f"High disagreement rate ({disagreement_rate:.0%})")
        return ". ".join(parts)

    def generate_adversarial_training_data(self) -> list[AdversarialExample]:
        """Generate training examples targeting discovered weaknesses.

        Strategy: For each weakness, take the best synthesized answer
        and pair it with the query. This teaches ReasoningBox to handle
        exactly the cases where other models fail.
        """
        data = self.load_data()
        weaknesses = self.analyze_weaknesses()
        weakness_categories = {w.category for w in weaknesses}

        adversarial_examples = []
        for entry in data:
            categories = self.categorize_query(entry["query"])

            # Focus on entries that hit weakness categories
            matching_weaknesses = [c for c in categories if c in weakness_categories]
            if not matching_weaknesses:
                continue

            # Only use high-quality synthesized answers
            if entry.get("confidence", 0) < 0.5:
                continue
            if not entry.get("final_answer"):
                continue

            # Find which models failed
            failed = []
            for resp in entry.get("responses", []):
                if resp.get("total_score", 0) < 0.5 and not resp.get("error"):
                    failed.append({
                        "model": resp.get("model_name", "unknown"),
                        "score": resp.get("total_score", 0),
                        "answer_preview": resp.get("content", "")[:200],
                    })

            adversarial_examples.append(AdversarialExample(
                query=entry["query"],
                best_answer=entry["final_answer"],
                weakness_category=matching_weaknesses[0],
                target_score_improvement=1.0 - entry.get("confidence", 0.5),
                failed_model_answers=failed,
            ))

        return adversarial_examples

    def export_adversarial_dataset(self, output_path: str | None = None) -> str:
        """Export adversarial training data as JSONL."""
        examples = self.generate_adversarial_training_data()
        if not examples:
            return "No adversarial examples generated. Need more ensemble data."

        path = Path(output_path) if output_path else self.data_dir / "adversarial_training.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

        system = (
            "You are ReasoningBox, an AI that excels at questions where other AI models fail. "
            "You have been specifically trained on cases involving complex reasoning, ambiguity, "
            "mathematical precision, and nuanced analysis. Be thorough and precise."
        )

        with open(path, "w") as f:
            for ex in examples:
                entry = {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": ex.query},
                        {"role": "assistant", "content": ex.best_answer},
                    ],
                    "metadata": {
                        "weakness_category": ex.weakness_category,
                        "target_improvement": ex.target_score_improvement,
                        "num_failed_models": len(ex.failed_model_answers),
                    },
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return f"Exported {len(examples)} adversarial examples to {path}"

    def get_weakness_report(self) -> dict:
        """Generate a full weakness analysis report."""
        weaknesses = self.analyze_weaknesses()
        adversarial = self.generate_adversarial_training_data()

        return {
            "total_weaknesses_found": len(weaknesses),
            "adversarial_examples_available": len(adversarial),
            "weaknesses": [
                {
                    "category": w.category,
                    "description": w.description,
                    "severity": round(w.severity, 3),
                    "avg_score": round(w.avg_score, 3),
                    "disagreement_rate": round(w.disagreement_rate, 3),
                    "affected_models": w.affected_models,
                    "example_count": w.count,
                    "sample_queries": w.example_queries[:3],
                }
                for w in weaknesses
            ],
            "top_vulnerability": weaknesses[0].category if weaknesses else None,
            "recommendation": self._get_recommendation(weaknesses),
        }

    def _get_recommendation(self, weaknesses: list[WeaknessPattern]) -> str:
        if not weaknesses:
            return "Need more data. Keep using the ensemble to collect training examples."
        top = weaknesses[0]
        if top.severity > 0.7:
            return (f"Critical weakness in '{top.category}'. Frontier models avg {top.avg_score:.0%} "
                    f"here. Training ReasoningBox on these examples could yield significant advantage.")
        elif top.severity > 0.4:
            return (f"Notable weakness in '{top.category}'. Focus adversarial training here "
                    f"for best ROI on model improvement.")
        else:
            return "Models are performing reasonably. Focus on edge cases and high-disagreement queries."
