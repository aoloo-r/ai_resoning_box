"""Google (Gemini) provider."""

from __future__ import annotations
import os
from core.models import ModelResponse
from core.providers.base import BaseProvider


class GoogleProvider(BaseProvider):
    name = "google"

    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        self.genai = genai

    async def generate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        import asyncio

        generation_config = self.genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        model = self.genai.GenerativeModel(
            model_name=model_id,
            system_instruction=system_prompt if system_prompt else None,
            generation_config=generation_config,
        )

        # google-generativeai is sync, run in executor
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: model.generate_content(prompt)
        )

        thinking_text = None
        content_text = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "thought") and part.thought:
                    thinking_text = part.text
                else:
                    content_text += part.text

        usage = {}
        if response.usage_metadata:
            usage = {
                "input": response.usage_metadata.prompt_token_count or 0,
                "output": response.usage_metadata.candidates_token_count or 0,
            }

        return ModelResponse(
            model_id=model_id,
            model_name="",
            provider=self.name,
            content=content_text,
            reasoning_trace=thinking_text,
            token_usage=usage,
        )
