"""Anthropic (Claude) provider."""

from __future__ import annotations
import os
from core.models import ModelResponse
from core.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None):
        import anthropic
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
        )

    async def generate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        kwargs: dict = {
            "model": model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        # Use extended thinking for opus models
        thinking_text = None
        if "opus" in model_id:
            kwargs.pop("temperature", None)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

        response = await self.client.messages.create(**kwargs)

        content_text = ""
        for block in response.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                content_text = block.text

        return ModelResponse(
            model_id=model_id,
            model_name="",
            provider=self.name,
            content=content_text,
            reasoning_trace=thinking_text,
            token_usage={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        )
