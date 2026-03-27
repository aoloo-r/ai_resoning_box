"""Data collector: saves ensemble results as training data for the ReasoningBox model."""

from __future__ import annotations
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from core.models import SynthesisResult

# Default data directory
DATA_DIR = Path(__file__).parent.parent / "training" / "data"


class DataCollector:
    """Collects ensemble results and saves them as JSONL training data."""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_file = self.data_dir / "raw_ensemble.jsonl"
        self.training_file = self.data_dir / "training_pairs.jsonl"
        self.stats_file = self.data_dir / "stats.json"

    def collect(self, result: SynthesisResult) -> dict:
        """Save a full ensemble result as raw data + extract training pair."""
        raw_entry = self._to_raw(result)
        training_entry = self._to_training_pair(result)

        # Append raw ensemble data
        self._append_jsonl(self.raw_file, raw_entry)

        # Append training pair (instruction -> best answer)
        if training_entry:
            self._append_jsonl(self.training_file, training_entry)

        # Update stats
        self._update_stats(result)

        return raw_entry

    def _to_raw(self, result: SynthesisResult) -> dict:
        """Convert full result to raw JSONL entry."""
        return {
            "id": result.id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": result.query,
            "strategy": result.strategy.value,
            "confidence": result.confidence,
            "final_answer": result.final_answer,
            "reasoning": result.reasoning,
            "consensus_points": result.consensus_points,
            "disagreement_points": result.disagreement_points,
            "total_latency_ms": result.total_latency_ms,
            "responses": [
                {
                    "model_id": s.response.model_id,
                    "model_name": s.response.model_name,
                    "provider": s.response.provider,
                    "content": s.response.content,
                    "reasoning_trace": s.response.reasoning_trace,
                    "latency_ms": s.response.latency_ms,
                    "error": s.response.error,
                    "scores": s.scores,
                    "total_score": s.total_score,
                    "rank": s.rank,
                }
                for s in result.individual_responses
            ],
        }

    def _to_training_pair(self, result: SynthesisResult) -> dict | None:
        """Extract a high-quality instruction/output training pair.

        Only saves pairs where:
        - Confidence >= 0.6
        - At least 2 model responses
        - Final answer is non-empty
        """
        if result.confidence < 0.6:
            return None
        if len(result.individual_responses) < 2:
            return None
        if not result.final_answer or len(result.final_answer.strip()) < 50:
            return None

        # Get the best individual response for comparison
        best = max(result.individual_responses, key=lambda s: s.total_score)

        # Build system prompt that teaches the model to reason like the ensemble
        system = (
            "You are ReasoningBox, an AI assistant trained on synthesized answers from "
            "multiple world-class AI models. You combine the best reasoning patterns from "
            "Claude, GPT-4, Gemini, and other frontier models. Provide thorough, accurate, "
            "and well-structured answers. Show your reasoning process clearly."
        )

        return {
            "id": result.id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": system,
            "instruction": result.query,
            "output": result.final_answer,
            "metadata": {
                "confidence": result.confidence,
                "strategy": result.strategy.value,
                "num_models": len(result.individual_responses),
                "best_model": best.response.model_name,
                "best_model_score": best.total_score,
                "consensus_count": len(result.consensus_points),
                "disagreement_count": len(result.disagreement_points),
            },
        }

    def _append_jsonl(self, filepath: Path, entry: dict):
        """Append a JSON entry to a JSONL file."""
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _update_stats(self, result: SynthesisResult):
        """Update running statistics."""
        stats = self.get_stats()

        stats["total_queries"] = stats.get("total_queries", 0) + 1
        stats["total_responses"] = stats.get("total_responses", 0) + len(result.individual_responses)
        stats["last_updated"] = datetime.now(timezone.utc).isoformat()

        # Track training-eligible count
        pair = self._to_training_pair(result)
        if pair:
            stats["training_pairs"] = stats.get("training_pairs", 0) + 1

        # Track average confidence
        total_conf = stats.get("_total_confidence", 0.0) + result.confidence
        stats["_total_confidence"] = total_conf
        stats["avg_confidence"] = total_conf / stats["total_queries"]

        # Track strategy usage
        strat_counts = stats.get("strategy_counts", {})
        strat_counts[result.strategy.value] = strat_counts.get(result.strategy.value, 0) + 1
        stats["strategy_counts"] = strat_counts

        # Track model usage
        model_counts = stats.get("model_counts", {})
        for s in result.individual_responses:
            name = s.response.model_name
            model_counts[name] = model_counts.get(name, 0) + 1
        stats["model_counts"] = model_counts

        # Track model win rates (rank #1)
        win_counts = stats.get("model_wins", {})
        for s in result.individual_responses:
            if s.rank == 1:
                name = s.response.model_name
                win_counts[name] = win_counts.get(name, 0) + 1
        stats["model_wins"] = win_counts

        # Dataset file sizes
        stats["raw_file_mb"] = round(self.raw_file.stat().st_size / (1024 * 1024), 2) if self.raw_file.exists() else 0
        stats["training_file_mb"] = round(self.training_file.stat().st_size / (1024 * 1024), 2) if self.training_file.exists() else 0

        # Estimate training readiness
        tp = stats.get("training_pairs", 0)
        if tp >= 1000:
            stats["training_readiness"] = "ready"
        elif tp >= 500:
            stats["training_readiness"] = "almost_ready"
        elif tp >= 100:
            stats["training_readiness"] = "collecting"
        else:
            stats["training_readiness"] = "early"

        stats["training_readiness_pct"] = min(100, round((tp / 1000) * 100, 1))

        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    def get_stats(self) -> dict:
        """Load current stats."""
        if self.stats_file.exists():
            with open(self.stats_file) as f:
                return json.load(f)
        return {}

    def get_recent_entries(self, n: int = 10) -> list[dict]:
        """Get the N most recent raw entries."""
        if not self.raw_file.exists():
            return []
        entries = []
        with open(self.raw_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries[-n:]

    def export_for_training(self, output_path: str | None = None) -> str:
        """Export training data in HuggingFace-compatible format."""
        output = Path(output_path) if output_path else self.data_dir / "hf_dataset.jsonl"

        if not self.training_file.exists():
            raise FileNotFoundError("No training data collected yet.")

        count = 0
        with open(self.training_file) as fin, open(output, "w") as fout:
            for line in fin:
                entry = json.loads(line.strip())
                # Convert to ChatML / ShareGPT format
                hf_entry = {
                    "conversations": [
                        {"role": "system", "content": entry["system"]},
                        {"role": "user", "content": entry["instruction"]},
                        {"role": "assistant", "content": entry["output"]},
                    ],
                    "metadata": entry.get("metadata", {}),
                }
                fout.write(json.dumps(hf_entry, ensure_ascii=False) + "\n")
                count += 1

        return f"Exported {count} training examples to {output}"
