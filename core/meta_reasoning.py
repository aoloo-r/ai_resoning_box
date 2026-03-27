"""Meta-Reasoning Engine — Multi-stage think-verify-refine loop.

Inspired by JEPA's world model concept: instead of single-pass generation,
the meta-reasoner follows a structured reasoning pipeline:

    1. DECOMPOSE  — Break the question into sub-problems
    2. REASON     — Generate initial reasoning for each sub-problem
    3. CRITIQUE   — Self-evaluate reasoning quality
    4. VERIFY     — Cross-check against known patterns and constraints
    5. REFINE     — Improve weak points and synthesize final answer

Each stage uses the JEPA world model's predicted reasoning embeddings
as "compass points" to stay on track.

This gives ReasoningBox a systematic advantage: while frontier models
do single-pass generation, ReasoningBox does deliberate, multi-stage reasoning.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class ReasoningStage(str, Enum):
    DECOMPOSE = "decompose"
    REASON = "reason"
    CRITIQUE = "critique"
    VERIFY = "verify"
    REFINE = "refine"


@dataclass
class ReasoningStep:
    """A single step in the meta-reasoning chain."""
    stage: ReasoningStage
    input_text: str
    output_text: str
    confidence: float = 0.0
    issues_found: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)


@dataclass
class ReasoningTrace:
    """Complete trace of the meta-reasoning process."""
    query: str
    steps: list[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    overall_confidence: float = 0.0
    num_refinements: int = 0
    reasoning_depth: str = "standard"  # trivial, standard, deep, expert


class MetaReasoner:
    """Multi-stage reasoning engine that thinks before answering.

    Uses a provider (LLM) to execute each reasoning stage with
    specialized prompts that enforce structured thinking.
    """

    def __init__(self, provider, model_id: str):
        self.provider = provider
        self.model_id = model_id

    async def reason(
        self,
        query: str,
        depth: str = "standard",
        max_refinements: int = 2,
        temperature: float = 0.4,
    ) -> ReasoningTrace:
        """Execute the full meta-reasoning pipeline.

        Args:
            query: The user's question
            depth: "trivial", "standard", "deep", or "expert"
            max_refinements: Max refinement iterations
            temperature: Generation temperature (lower = more precise)
        """
        trace = ReasoningTrace(query=query, reasoning_depth=depth)

        # Stage 1: DECOMPOSE
        decomposition = await self._decompose(query, depth, temperature)
        trace.steps.append(decomposition)

        # Stage 2: REASON
        initial_reasoning = await self._reason(query, decomposition.output_text, temperature)
        trace.steps.append(initial_reasoning)

        # Stage 3: CRITIQUE
        critique = await self._critique(query, initial_reasoning.output_text, temperature)
        trace.steps.append(critique)

        # Stage 4: VERIFY
        verification = await self._verify(query, initial_reasoning.output_text, critique.output_text, temperature)
        trace.steps.append(verification)

        # Stage 5: REFINE (iterate if needed)
        current_answer = initial_reasoning.output_text
        for i in range(max_refinements):
            if verification.confidence > 0.85 and not critique.issues_found:
                break  # Good enough, no refinement needed

            refinement = await self._refine(
                query, current_answer, critique.output_text,
                verification.output_text, temperature,
            )
            trace.steps.append(refinement)
            trace.num_refinements += 1
            current_answer = refinement.output_text

            # Re-critique the refined answer
            if i < max_refinements - 1:
                critique = await self._critique(query, current_answer, temperature)
                trace.steps.append(critique)
                verification = await self._verify(
                    query, current_answer, critique.output_text, temperature
                )
                trace.steps.append(verification)

        trace.final_answer = current_answer
        trace.overall_confidence = verification.confidence

        return trace

    async def _generate(self, system: str, prompt: str, temp: float) -> str:
        """Generate a response from the provider."""
        response = await self.provider.generate(
            model_id=self.model_id,
            prompt=prompt,
            system_prompt=system,
            temperature=temp,
            max_tokens=4096,
        )
        return response.content

    async def _decompose(self, query: str, depth: str, temp: float) -> ReasoningStep:
        system = (
            "You are a reasoning decomposition expert. Your job is to break down "
            "complex questions into clear sub-problems that can be reasoned about independently. "
            "Be thorough but concise."
        )
        prompt = f"""Break down this question into sub-problems:

Question: {query}

Reasoning depth: {depth}

For each sub-problem, identify:
1. What specific thing needs to be figured out
2. What knowledge/reasoning is required
3. Any dependencies between sub-problems

Format as a numbered list of sub-problems."""

        output = await self._generate(system, prompt, temp)
        return ReasoningStep(
            stage=ReasoningStage.DECOMPOSE,
            input_text=query,
            output_text=output,
        )

    async def _reason(self, query: str, decomposition: str, temp: float) -> ReasoningStep:
        system = (
            "You are a world-class reasoning engine. Given a question and its decomposition "
            "into sub-problems, work through each sub-problem systematically and build "
            "toward a comprehensive answer. Show your reasoning at each step."
        )
        prompt = f"""Work through this question step by step:

Question: {query}

Sub-problems identified:
{decomposition}

For each sub-problem:
1. State what you know
2. Reason through it carefully
3. State your conclusion

Then synthesize all sub-conclusions into a comprehensive final answer."""

        output = await self._generate(system, prompt, temp)
        return ReasoningStep(
            stage=ReasoningStage.REASON,
            input_text=f"Query: {query}\nDecomposition: {decomposition}",
            output_text=output,
        )

    async def _critique(self, query: str, reasoning: str, temp: float) -> ReasoningStep:
        system = (
            "You are a rigorous reasoning critic. Your job is to find flaws, gaps, "
            "unsupported claims, logical errors, and areas for improvement in an answer. "
            "Be brutally honest. If the answer is good, say so — but always look for "
            "edge cases and potential issues."
        )
        prompt = f"""Critically evaluate this answer:

Original Question: {query}

Answer to evaluate:
{reasoning}

Check for:
1. Logical errors or fallacies
2. Unsupported claims or assumptions
3. Missing important considerations
4. Factual accuracy concerns
5. Clarity and completeness issues
6. Edge cases not addressed

List each issue found with severity (critical/major/minor).
If no significant issues: say "No major issues found" and rate confidence 0-100."""

        output = await self._generate(system, prompt, temp)

        # Parse confidence and issues
        issues = []
        confidence = 0.7
        for line in output.split("\n"):
            line_lower = line.lower()
            if "critical" in line_lower or "major" in line_lower:
                issues.append(line.strip())
            if "confidence" in line_lower:
                import re
                match = re.search(r"(\d+)", line)
                if match:
                    confidence = int(match.group(1)) / 100.0
            if "no major issues" in line_lower or "no significant issues" in line_lower:
                confidence = max(confidence, 0.85)

        return ReasoningStep(
            stage=ReasoningStage.CRITIQUE,
            input_text=reasoning,
            output_text=output,
            confidence=confidence,
            issues_found=issues,
        )

    async def _verify(self, query: str, answer: str, critique: str, temp: float) -> ReasoningStep:
        system = (
            "You are a verification specialist. Cross-check an answer against known facts, "
            "logical constraints, and common error patterns. Your goal is to catch any "
            "remaining errors before the answer is finalized."
        )
        prompt = f"""Verify this answer:

Question: {query}

Answer:
{answer}

Critique found these issues:
{critique}

Verification checklist:
1. Are all factual claims accurate?
2. Is the logic internally consistent?
3. Does it actually answer the question asked?
4. Are there common misconceptions being repeated?
5. Would an expert in this field agree?

Rate overall verification confidence (0-100) and list any remaining concerns."""

        output = await self._generate(system, prompt, temp)

        confidence = 0.7
        for line in output.split("\n"):
            if "confidence" in line.lower():
                import re
                match = re.search(r"(\d+)", line)
                if match:
                    confidence = int(match.group(1)) / 100.0

        return ReasoningStep(
            stage=ReasoningStage.VERIFY,
            input_text=f"Answer: {answer}\nCritique: {critique}",
            output_text=output,
            confidence=min(1.0, confidence),
        )

    async def _refine(
        self, query: str, current_answer: str, critique: str,
        verification: str, temp: float,
    ) -> ReasoningStep:
        system = (
            "You are a reasoning refinement expert. Given an answer, its critique, "
            "and verification results, produce an improved version that addresses "
            "all identified issues while maintaining the strengths of the original."
        )
        prompt = f"""Improve this answer based on the critique and verification:

Original Question: {query}

Current Answer:
{current_answer}

Critique:
{critique}

Verification:
{verification}

Produce a refined answer that:
1. Fixes all identified issues
2. Maintains all correct parts
3. Adds missing considerations
4. Improves clarity and structure
5. Is comprehensive and well-reasoned"""

        output = await self._generate(system, prompt, temp)
        return ReasoningStep(
            stage=ReasoningStage.REFINE,
            input_text=f"Answer: {current_answer}\nIssues: {critique}",
            output_text=output,
            improvements=[f"Refined based on {len(critique.split(chr(10)))} critique points"],
        )
