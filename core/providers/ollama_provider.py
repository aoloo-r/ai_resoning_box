"""Ollama provider for local models."""

from __future__ import annotations
import os
from core.models import ModelResponse
from core.providers.base import BaseProvider


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self):
        import httpx
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.http = httpx.AsyncClient(timeout=300)

    async def generate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        payload = {
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        resp = await self.http.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()

        return ModelResponse(
            model_id=model_id,
            model_name="",
            provider=self.name,
            content=data.get("response", ""),
            token_usage={
                "input": data.get("prompt_eval_count", 0),
                "output": data.get("eval_count", 0),
            },
        )
