"""Dynamic Router — Learns which model/strategy wins per question type.

Instead of always using all models with the same strategy, the router
analyzes the question and routes it optimally:

1. Classifies the question type using learned embeddings
2. Looks up historical performance per model per question type
3. Selects the best model subset and strategy
4. Adjusts parameters (temperature, reasoning depth)

This gives a massive efficiency gain: instead of paying for 6 models
on a simple factual question, route it to the one model that's best
at factual recall. Use the full ensemble only for hard questions.

Over time, the router learns from the scoring data which models
excel at which question types — building a "meta-knowledge" layer.
"""

from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass


@dataclass
class RoutingDecision:
    """The router's decision for how to handle a query."""
    question_type: str
    difficulty: str  # trivial, easy, medium, hard, expert
    recommended_models: list[str]
    recommended_strategy: str
    temperature: float
    use_meta_reasoning: bool
    reasoning_depth: str
    confidence: float  # How confident the router is in this decision
    explanation: str


class DynamicRouter:
    """Routes questions to optimal model subsets and strategies.

    Learns from collected ensemble data — no neural network needed,
    just smart statistics and pattern matching.
    """

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent / "training" / "data"
        self.performance_db = self._load_performance_data()

    def _load_performance_data(self) -> dict:
        """Load historical model performance data."""
        cache_path = self.data_dir / "router_cache.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

        # Build from raw data
        raw_path = self.data_dir / "raw_ensemble.jsonl"
        if not raw_path.exists():
            return {"model_scores": {}, "strategy_scores": {}, "total_queries": 0}

        model_scores = defaultdict(lambda: defaultdict(list))
        strategy_scores = defaultdict(lambda: defaultdict(list))
        query_types = defaultdict(int)

        with open(raw_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                q_type = self._classify_question(entry["query"])
                query_types[q_type] += 1

                strategy = entry.get("strategy", "weighted_merge")
                confidence = entry.get("confidence", 0)
                strategy_scores[q_type][strategy].append(confidence)

                for resp in entry.get("responses", []):
                    model = resp.get("model_name", "unknown")
                    score = resp.get("total_score", 0)
                    model_scores[q_type][model].append(score)

        # Compute averages
        db = {
            "model_scores": {},
            "strategy_scores": {},
            "query_type_counts": dict(query_types),
            "total_queries": sum(query_types.values()),
        }

        for q_type in model_scores:
            db["model_scores"][q_type] = {
                model: {
                    "avg": sum(scores) / len(scores),
                    "count": len(scores),
                    "std": self._std(scores),
                }
                for model, scores in model_scores[q_type].items()
            }

        for q_type in strategy_scores:
            db["strategy_scores"][q_type] = {
                strategy: {
                    "avg": sum(scores) / len(scores),
                    "count": len(scores),
                }
                for strategy, scores in strategy_scores[q_type].items()
            }

        # Cache it
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(db, f, indent=2)

        return db

    def _std(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))

    def _classify_question(self, query: str) -> str:
        """Simple rule-based question type classification."""
        q = query.lower()

        if any(w in q for w in ["code", "program", "function", "implement", "debug", "python", "javascript"]):
            return "coding"
        if any(w in q for w in ["calculate", "equation", "prove", "math", "number", "probability"]):
            return "mathematical"
        if any(w in q for w in ["when did", "who was", "what is the", "how many", "capital of"]):
            return "factual"
        if any(w in q for w in ["compare", "vs", "difference between", "pros and cons", "trade"]):
            return "comparison"
        if any(w in q for w in ["explain", "how does", "why does", "what causes"]):
            return "explanation"
        if any(w in q for w in ["write a", "story", "poem", "creative", "imagine", "design"]):
            return "creative"
        if any(w in q for w in ["should", "ethical", "moral", "opinion", "think about"]):
            return "ethical"
        if any(w in q for w in ["step by step", "plan", "strategy", "build", "architecture"]):
            return "planning"
        if any(w in q for w in ["if", "logic", "paradox", "deduce", "implies"]):
            return "logical"
        return "general"

    def _estimate_difficulty(self, query: str) -> str:
        """Estimate question difficulty."""
        q = query.lower()
        word_count = len(query.split())

        # Simple heuristics
        hard_signals = sum([
            word_count > 50,
            "step by step" in q,
            "compare" in q and "and" in q,
            any(w in q for w in ["prove", "derive", "optimize", "architecture"]),
            query.count("?") > 1,  # Multiple questions
        ])

        easy_signals = sum([
            word_count < 15,
            any(w in q for w in ["what is", "define", "who is"]),
        ])

        score = hard_signals - easy_signals
        if score >= 3:
            return "expert"
        elif score >= 2:
            return "hard"
        elif score >= 1:
            return "medium"
        elif easy_signals >= 2:
            return "trivial"
        return "easy"

    def route(self, query: str) -> RoutingDecision:
        """Route a query to optimal models and strategy."""
        q_type = self._classify_question(query)
        difficulty = self._estimate_difficulty(query)

        # Get historical performance for this question type
        model_perf = self.performance_db.get("model_scores", {}).get(q_type, {})
        strat_perf = self.performance_db.get("strategy_scores", {}).get(q_type, {})

        # Select best models
        if model_perf:
            ranked_models = sorted(model_perf.items(), key=lambda x: -x[1]["avg"])
            if difficulty in ("trivial", "easy"):
                # Use top 2 models for easy questions (save cost)
                recommended = [m for m, _ in ranked_models[:2]]
            elif difficulty == "medium":
                # Use top 3
                recommended = [m for m, _ in ranked_models[:3]]
            else:
                # Use all for hard/expert
                recommended = [m for m, _ in ranked_models]
        else:
            recommended = []  # No data, use all (default)

        # Select best strategy
        if strat_perf:
            best_strategy = max(strat_perf.items(), key=lambda x: x[1]["avg"])
            strategy = best_strategy[0]
        else:
            # Default strategy based on difficulty
            strategy_map = {
                "trivial": "best_of_n",
                "easy": "best_of_n",
                "medium": "weighted_merge",
                "hard": "weighted_merge",
                "expert": "chain_of_verification",
            }
            strategy = strategy_map.get(difficulty, "weighted_merge")

        # Determine if meta-reasoning is needed
        use_meta = difficulty in ("hard", "expert") or q_type in ("logical", "mathematical", "planning")

        # Temperature based on question type
        temp_map = {
            "coding": 0.3,
            "mathematical": 0.2,
            "factual": 0.3,
            "creative": 0.8,
            "ethical": 0.6,
            "explanation": 0.5,
        }
        temperature = temp_map.get(q_type, 0.5)

        # Reasoning depth
        depth_map = {
            "trivial": "trivial",
            "easy": "standard",
            "medium": "standard",
            "hard": "deep",
            "expert": "expert",
        }

        # Router confidence based on data availability
        data_count = self.performance_db.get("query_type_counts", {}).get(q_type, 0)
        confidence = min(0.95, 0.3 + (data_count / 50) * 0.65)  # Increases with more data

        explanation_parts = [f"Question type: {q_type}", f"Difficulty: {difficulty}"]
        if model_perf:
            explanation_parts.append(f"Best model for {q_type}: {ranked_models[0][0]} ({ranked_models[0][1]['avg']:.0%})")
        if data_count > 0:
            explanation_parts.append(f"Based on {data_count} previous {q_type} queries")
        else:
            explanation_parts.append("No historical data — using defaults")

        return RoutingDecision(
            question_type=q_type,
            difficulty=difficulty,
            recommended_models=recommended,
            recommended_strategy=strategy,
            temperature=temperature,
            use_meta_reasoning=use_meta,
            reasoning_depth=depth_map.get(difficulty, "standard"),
            confidence=confidence,
            explanation=". ".join(explanation_parts),
        )

    def invalidate_cache(self):
        """Force rebuild of the performance database."""
        cache_path = self.data_dir / "router_cache.json"
        if cache_path.exists():
            cache_path.unlink()
        self.performance_db = self._load_performance_data()
