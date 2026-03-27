"""Data models for the AI Ensemble platform."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time
import uuid


class SynthesisStrategy(str, Enum):
    BEST_OF_N = "best_of_n"
    WEIGHTED_MERGE = "weighted_merge"
    DEBATE = "debate"
    CHAIN_OF_VERIFICATION = "chain_of_verification"


class ModelRole(str, Enum):
    REASONING = "reasoning"
    FAST = "fast"
    DEEP_REASONING = "deep_reasoning"


@dataclass
class ModelConfig:
    id: str
    name: str
    provider: str
    role: ModelRole = ModelRole.REASONING


@dataclass
class ModelResponse:
    model_id: str
    model_name: str
    provider: str
    content: str
    reasoning_trace: str | None = None
    latency_ms: float = 0
    token_usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredResponse:
    response: ModelResponse
    scores: dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    rank: int = 0


@dataclass
class SynthesisResult:
    """Final synthesized output from all models."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = ""
    strategy: SynthesisStrategy = SynthesisStrategy.WEIGHTED_MERGE
    final_answer: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    individual_responses: list[ScoredResponse] = field(default_factory=list)
    consensus_points: list[str] = field(default_factory=list)
    disagreement_points: list[str] = field(default_factory=list)
    total_latency_ms: float = 0
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        lines = [
            f"Query: {self.query[:100]}...",
            f"Strategy: {self.strategy.value}",
            f"Models used: {len(self.individual_responses)}",
            f"Confidence: {self.confidence:.0%}",
            f"Total latency: {self.total_latency_ms:.0f}ms",
        ]
        if self.consensus_points:
            lines.append(f"Consensus on {len(self.consensus_points)} points")
        if self.disagreement_points:
            lines.append(f"Disagreements on {len(self.disagreement_points)} points")
        return "\n".join(lines)
