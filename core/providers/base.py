"""Abstract base provider for AI model integrations."""

from __future__ import annotations
from abc import ABC, abstractmethod
from core.models import ModelResponse
import time


class BaseProvider(ABC):
    """Base class all AI providers must implement."""

    name: str = "base"

    @abstractmethod
    async def generate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        ...

    async def timed_generate(
        self,
        model_id: str,
        model_name: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        start = time.perf_counter()
        try:
            response = await self.generate(
                model_id=model_id,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response.latency_ms = (time.perf_counter() - start) * 1000
            response.model_name = model_name
            return response
        except Exception as e:
            return ModelResponse(
                model_id=model_id,
                model_name=model_name,
                provider=self.name,
                content="",
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000,
            )
