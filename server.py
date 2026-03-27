#!/usr/bin/env python3
"""AI Ensemble Platform — FastAPI Server.

Run:
    uvicorn server:app --reload --port 8900
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from core.models import SynthesisStrategy, ModelRole
from core.pipeline import EnsemblePipeline

app = FastAPI(
    title="AI Ensemble Reasoning Platform",
    description="Multi-model AI reasoning with synthesis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ApiKeys(BaseModel):
    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    DEEPSEEK_API_KEY: str | None = None


class QueryRequest(BaseModel):
    query: str
    system_prompt: str | None = None
    strategy: str = "weighted_merge"
    roles: list[str] | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    api_keys: ApiKeys | None = None


class StatusRequest(BaseModel):
    api_keys: ApiKeys | None = None


class ModelResponseOut(BaseModel):
    model_id: str
    model_name: str
    provider: str
    content: str
    reasoning_trace: str | None = None
    latency_ms: float
    error: str | None = None
    score: float = 0.0
    rank: int = 0
    scores: dict = {}


class SynthesisResultOut(BaseModel):
    id: str
    query: str
    strategy: str
    final_answer: str
    confidence: float
    reasoning: str
    consensus_points: list[str]
    disagreement_points: list[str]
    total_latency_ms: float
    individual_responses: list[ModelResponseOut]


def _build_keys_dict(api_keys: ApiKeys | None) -> dict[str, str]:
    """Convert ApiKeys model to a plain dict, filtering out empty values."""
    if not api_keys:
        return {}
    keys = {}
    for field in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY"]:
        val = getattr(api_keys, field, None)
        if val and val.strip():
            keys[field] = val.strip()
    return keys


def _make_pipeline(api_keys: ApiKeys | None = None) -> EnsemblePipeline:
    """Create a pipeline instance with user-provided API keys."""
    keys = _build_keys_dict(api_keys)
    return EnsemblePipeline(api_keys=keys)


@app.post("/api/status")
async def get_status_post(req: StatusRequest):
    """Get status using user-provided API keys."""
    pipeline = _make_pipeline(req.api_keys)
    return pipeline.get_status()


@app.get("/api/status")
async def get_status():
    """Get status using env-based API keys (fallback)."""
    pipeline = _make_pipeline()
    return pipeline.get_status()


@app.post("/api/query", response_model=SynthesisResultOut)
async def query(req: QueryRequest):
    pipeline = _make_pipeline(req.api_keys)

    try:
        strategy = SynthesisStrategy(req.strategy)
    except ValueError:
        raise HTTPException(400, f"Invalid strategy: {req.strategy}")

    roles = None
    if req.roles:
        try:
            roles = [ModelRole(r) for r in req.roles]
        except ValueError as e:
            raise HTTPException(400, str(e))

    result = await pipeline.run(
        query=req.query,
        system_prompt=req.system_prompt,
        strategy=strategy,
        roles=roles,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    return SynthesisResultOut(
        id=result.id,
        query=result.query,
        strategy=result.strategy.value,
        final_answer=result.final_answer,
        confidence=result.confidence,
        reasoning=result.reasoning,
        consensus_points=result.consensus_points,
        disagreement_points=result.disagreement_points,
        total_latency_ms=result.total_latency_ms,
        individual_responses=[
            ModelResponseOut(
                model_id=s.response.model_id,
                model_name=s.response.model_name,
                provider=s.response.provider,
                content=s.response.content,
                reasoning_trace=s.response.reasoning_trace,
                latency_ms=s.response.latency_ms,
                error=s.response.error,
                score=s.total_score,
                rank=s.rank,
                scores=s.scores,
            )
            for s in result.individual_responses
        ],
    )


@app.get("/api/strategies")
async def get_strategies():
    return [{"id": s.value, "name": s.value.replace("_", " ").title()} for s in SynthesisStrategy]


# Serve static UI files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
