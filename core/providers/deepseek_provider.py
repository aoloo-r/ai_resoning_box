"""DeepSeek provider (OpenAI-compatible API)."""

from __future__ import annotations
import os
from core.models import ModelResponse
from core.providers.base import BaseProvider


class DeepSeekProvider(BaseProvider):
    name = "deepseek"

    def __init__(self, api_key: str | None = None):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
        )

    async def generate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        reasoning = None
        if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
            reasoning = choice.message.reasoning_content

        return ModelResponse(
            model_id=model_id,
            model_name="",
            provider=self.name,
            content=choice.message.content or "",
            reasoning_trace=reasoning,
            token_usage={
                "input": response.usage.prompt_tokens if response.usage else 0,
                "output": response.usage.completion_tokens if response.usage else 0,
            },
        )
