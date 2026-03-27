"""Synthesizer: evaluates, scores, and merges multi-model responses into a final answer."""

from __future__ import annotations
import json
from core.models import (
    ModelResponse,
    ScoredResponse,
    SynthesisResult,
    SynthesisStrategy,
)
from core.providers.base import BaseProvider


SCORING_PROMPT = """You are an expert AI response evaluator. You will be given a user query and multiple AI model responses. Your job is to:

1. Score each response on these dimensions (0.0 to 1.0):
   - accuracy: factual correctness and precision
   - completeness: how thoroughly the question is answered
   - reasoning_quality: logical coherence, depth of analysis
   - clarity: readability, structure, conciseness

2. Identify consensus points (things most/all models agree on)
3. Identify disagreement points (where models diverge)
4. Determine an overall confidence level (0.0 to 1.0)

Respond in this exact JSON format:
{
  "scores": [
    {"model_index": 0, "accuracy": 0.9, "completeness": 0.8, "reasoning_quality": 0.85, "clarity": 0.9},
    ...
  ],
  "consensus_points": ["point 1", "point 2"],
  "disagreement_points": ["disagreement 1"],
  "confidence": 0.85
}

USER QUERY:
{{QUERY}}

MODEL RESPONSES:
{{RESPONSES}}"""

MERGE_PROMPT = """You are an expert synthesizer. Given a user query and multiple AI model responses with their quality scores, produce the BEST POSSIBLE answer by:

1. Taking the strongest elements from each response
2. Resolving disagreements by favoring the most well-reasoned positions
3. Filling gaps where one model covered something others missed
4. Ensuring factual accuracy — if models disagree on facts, flag uncertainty
5. Maintaining clear, well-structured output

Do NOT mention the individual models or that this is a synthesis. Just produce the best answer as if you were the single, most knowledgeable expert.

USER QUERY:
{{QUERY}}

MODEL RESPONSES (ranked by quality score):
{{RANKED_RESPONSES}}

CONSENSUS POINTS:
{{CONSENSUS}}

DISAGREEMENT POINTS:
{{DISAGREEMENTS}}

Produce the definitive answer:"""

DEBATE_PROMPT = """You are model {{MODEL_NAME}}. Another model responded to the same query differently from you. Review their response and either:
1. Update your answer if they made valid points you missed
2. Defend your position if you believe you are more correct
3. Note where you now agree and where you still disagree

YOUR ORIGINAL RESPONSE:
{{YOUR_RESPONSE}}

OTHER MODEL'S RESPONSE:
{{OTHER_RESPONSE}}

Provide your updated, refined response:"""

VERIFY_PROMPT = """You are a verification expert. Given a synthesized answer, check it for:
1. Factual errors or unsupported claims
2. Logical inconsistencies
3. Missing important caveats or nuances
4. Overconfident statements that should be hedged

If you find issues, provide a corrected version. If the answer is solid, return it unchanged with a brief note on why it's reliable.

ORIGINAL QUERY: {{QUERY}}
SYNTHESIZED ANSWER: {{ANSWER}}

Your verified response:"""


class Synthesizer:
    """Evaluates and synthesizes multi-model responses."""

    def __init__(self, judge_provider: BaseProvider, judge_model: str):
        self.judge = judge_provider
        self.judge_model = judge_model

    async def _judge_call(self, prompt: str, max_tokens: int = 4096) -> str:
        resp = await self.judge.timed_generate(
            model_id=self.judge_model,
            model_name="judge",
            prompt=prompt,
            system_prompt="You are a precise evaluator. Always respond in valid JSON when asked for JSON.",
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return resp.content

    async def score_responses(
        self, query: str, responses: list[ModelResponse]
    ) -> tuple[list[ScoredResponse], list[str], list[str], float]:
        """Score all responses and identify consensus/disagreements."""
        response_text = ""
        for i, r in enumerate(responses):
            if r.error:
                response_text += f"\n--- Model {i} ({r.model_name}, {r.provider}) [ERROR: {r.error}] ---\n"
                continue
            response_text += f"\n--- Model {i} ({r.model_name}, {r.provider}) ---\n{r.content}\n"

        prompt = SCORING_PROMPT.replace("{{QUERY}}", query).replace("{{RESPONSES}}", response_text)
        raw = await self._judge_call(prompt)

        # Parse JSON from response
        scores_data = self._parse_json(raw)

        scored = []
        weights = {"accuracy": 0.35, "completeness": 0.25, "reasoning_quality": 0.20, "clarity": 0.10}
        consensus_weight = 0.10

        for i, r in enumerate(responses):
            model_scores = {}
            total = 0.0
            if scores_data and "scores" in scores_data:
                for s in scores_data["scores"]:
                    if s.get("model_index") == i:
                        model_scores = {
                            "accuracy": s.get("accuracy", 0),
                            "completeness": s.get("completeness", 0),
                            "reasoning_quality": s.get("reasoning_quality", 0),
                            "clarity": s.get("clarity", 0),
                        }
                        for dim, w in weights.items():
                            total += model_scores.get(dim, 0) * w
                        break

            scored.append(ScoredResponse(
                response=r,
                scores=model_scores,
                total_score=total,
            ))

        # Rank
        scored.sort(key=lambda x: x.total_score, reverse=True)
        for i, s in enumerate(scored):
            s.rank = i + 1

        consensus = scores_data.get("consensus_points", []) if scores_data else []
        disagreements = scores_data.get("disagreement_points", []) if scores_data else []
        confidence = scores_data.get("confidence", 0.5) if scores_data else 0.5

        return scored, consensus, disagreements, confidence

    async def weighted_merge(
        self, query: str, scored: list[ScoredResponse],
        consensus: list[str], disagreements: list[str]
    ) -> str:
        """Merge responses weighted by quality scores."""
        ranked_text = ""
        for s in scored:
            if s.response.error:
                continue
            ranked_text += (
                f"\n--- {s.response.model_name} ({s.response.provider}) "
                f"[Score: {s.total_score:.2f}, Rank: {s.rank}] ---\n"
                f"{s.response.content}\n"
            )

        prompt = (
            MERGE_PROMPT
            .replace("{{QUERY}}", query)
            .replace("{{RANKED_RESPONSES}}", ranked_text)
            .replace("{{CONSENSUS}}", "\n".join(f"- {c}" for c in consensus) or "None identified")
            .replace("{{DISAGREEMENTS}}", "\n".join(f"- {d}" for d in disagreements) or "None identified")
        )

        return await self._judge_call(prompt, max_tokens=8192)

    async def best_of_n(self, scored: list[ScoredResponse]) -> str:
        """Simply return the highest-scored response."""
        valid = [s for s in scored if not s.response.error]
        if not valid:
            return "All models failed to produce a response."
        return valid[0].response.content

    async def debate(
        self, query: str, responses: list[ModelResponse],
        orchestrator, rounds: int = 2
    ) -> list[ModelResponse]:
        """Run debate rounds where models critique each other's responses."""
        current = list(responses)
        valid = [r for r in current if not r.error]
        if len(valid) < 2:
            return current

        for _round in range(rounds):
            updated = []
            for i, resp in enumerate(valid):
                # Pick the highest-quality other response to debate against
                other = valid[(i + 1) % len(valid)]
                debate_prompt = (
                    DEBATE_PROMPT
                    .replace("{{MODEL_NAME}}", resp.model_name)
                    .replace("{{YOUR_RESPONSE}}", resp.content)
                    .replace("{{OTHER_RESPONSE}}", other.content)
                )
                # Use the same model for its own debate response
                model_config = None
                for m in orchestrator.models:
                    if m.id == resp.model_id:
                        model_config = m
                        break
                if model_config:
                    new_resp = await orchestrator.query_model(model_config, debate_prompt)
                    if not new_resp.error:
                        updated.append(new_resp)
                    else:
                        updated.append(resp)
                else:
                    updated.append(resp)
            valid = updated
        return valid

    async def verify(self, query: str, answer: str) -> str:
        """Run chain-of-verification on the final answer."""
        prompt = VERIFY_PROMPT.replace("{{QUERY}}", query).replace("{{ANSWER}}", answer)
        return await self._judge_call(prompt, max_tokens=8192)

    async def synthesize(
        self,
        query: str,
        responses: list[ModelResponse],
        strategy: SynthesisStrategy = SynthesisStrategy.WEIGHTED_MERGE,
        orchestrator=None,
        debate_rounds: int = 2,
    ) -> SynthesisResult:
        """Full synthesis pipeline."""
        import time
        start = time.perf_counter()

        # If debate, run debate rounds first
        working_responses = responses
        if strategy == SynthesisStrategy.DEBATE and orchestrator:
            working_responses = await self.debate(
                query, responses, orchestrator, debate_rounds
            )

        # Score all responses
        scored, consensus, disagreements, confidence = await self.score_responses(
            query, working_responses
        )

        # Generate final answer based on strategy
        if strategy == SynthesisStrategy.BEST_OF_N:
            final = await self.best_of_n(scored)
        elif strategy in (
            SynthesisStrategy.WEIGHTED_MERGE,
            SynthesisStrategy.DEBATE,
        ):
            final = await self.weighted_merge(query, scored, consensus, disagreements)
        elif strategy == SynthesisStrategy.CHAIN_OF_VERIFICATION:
            merged = await self.weighted_merge(query, scored, consensus, disagreements)
            final = await self.verify(query, merged)
        else:
            final = await self.best_of_n(scored)

        total_ms = (time.perf_counter() - start) * 1000

        return SynthesisResult(
            query=query,
            strategy=strategy,
            final_answer=final,
            confidence=confidence,
            reasoning=f"Synthesized from {len([s for s in scored if not s.response.error])} model responses using {strategy.value}",
            individual_responses=scored,
            consensus_points=consensus,
            disagreement_points=disagreements,
            total_latency_ms=total_ms + sum(r.latency_ms for r in responses),
        )

    def _parse_json(self, text: str) -> dict | None:
        """Extract JSON from model response text."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try to find JSON block in text
        import re
        patterns = [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
            r"(\{.*\})",
        ]
        for p in patterns:
            match = re.search(p, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        return None
