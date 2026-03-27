"""OpenAI (GPT-4, o1, o3) provider."""

from __future__ import annotations
import os
from core.models import ModelResponse
from core.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
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

        # o-series models use different params
        is_reasoning = model_id.startswith("o1") or model_id.startswith("o3")

        kwargs: dict = {
            "model": model_id,
            "messages": messages,
        }
        if is_reasoning:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens

        response = await self.client.chat.completions.create(**kwargs)

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
