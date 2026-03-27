"""ReasoningBox provider — serves the ensemble-trained fine-tuned model.

Supports three backends:
1. Local transformers (default) — loads the merged model directly
2. Ollama — if the model has been imported into Ollama
3. vLLM — for high-throughput production serving
"""

from __future__ import annotations
import os
from core.models import ModelResponse
from core.providers.base import BaseProvider


SYSTEM_PROMPT = (
    "You are ReasoningBox, an AI assistant trained on synthesized answers from "
    "multiple world-class AI models. You combine the best reasoning patterns from "
    "Claude, GPT-4, Gemini, and other frontier models. Provide thorough, accurate, "
    "and well-structured answers. Show your reasoning process clearly."
)


class ReasoningBoxProvider(BaseProvider):
    """Provider for the fine-tuned ReasoningBox model."""

    name = "reasoning_box"

    def __init__(self, model_path: str | None = None, backend: str = "ollama"):
        self.backend = backend
        self.model_path = model_path or os.environ.get(
            "REASONING_BOX_MODEL_PATH",
            "reasoning-box"
        )
        self._model = None
        self._tokenizer = None
        self._http = None

        if backend == "ollama":
            import httpx
            self.ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            self._http = httpx.AsyncClient(timeout=300)
        elif backend == "vllm":
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key="not-needed",
                base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            )

    def _load_local_model(self):
        """Lazy-load the local transformers model."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    async def generate(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        sys = system_prompt or SYSTEM_PROMPT

        if self.backend == "ollama":
            return await self._generate_ollama(model_id, prompt, sys, temperature, max_tokens)
        elif self.backend == "vllm":
            return await self._generate_vllm(model_id, prompt, sys, temperature, max_tokens)
        else:
            return await self._generate_local(model_id, prompt, sys, temperature, max_tokens)

    async def _generate_ollama(
        self, model_id: str, prompt: str, system_prompt: str,
        temperature: float, max_tokens: int,
    ) -> ModelResponse:
        payload = {
            "model": model_id,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        resp = await self._http.post(f"{self.ollama_url}/api/generate", json=payload)
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

    async def _generate_vllm(
        self, model_id: str, prompt: str, system_prompt: str,
        temperature: float, max_tokens: int,
    ) -> ModelResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = await self._client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        return ModelResponse(
            model_id=model_id,
            model_name="",
            provider=self.name,
            content=choice.message.content or "",
            token_usage={
                "input": response.usage.prompt_tokens if response.usage else 0,
                "output": response.usage.completion_tokens if response.usage else 0,
            },
        )

    async def _generate_local(
        self, model_id: str, prompt: str, system_prompt: str,
        temperature: float, max_tokens: int,
    ) -> ModelResponse:
        import asyncio
        import torch

        self._load_local_model()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        def _run():
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                )

            generated = outputs[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(generated, skip_special_tokens=True)

        content = await asyncio.get_event_loop().run_in_executor(None, _run)

        return ModelResponse(
            model_id=model_id,
            model_name="",
            provider=self.name,
            content=content,
        )
