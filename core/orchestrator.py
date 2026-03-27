"""Orchestrator: dispatches prompts to multiple AI models in parallel and collects responses."""

from __future__ import annotations
import asyncio
import os
import yaml
from pathlib import Path
from core.models import ModelConfig, ModelResponse, ModelRole
from core.providers.base import BaseProvider


class Orchestrator:
    """Manages providers and dispatches queries to multiple models concurrently."""

    def __init__(self, config_path: str | None = None, api_keys: dict[str, str] | None = None):
        self.providers: dict[str, BaseProvider] = {}
        self.models: list[ModelConfig] = []
        self.config: dict = {}
        self.api_keys = api_keys or {}
        self._load_config(config_path)

    def _load_config(self, config_path: str | None = None):
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        for provider_name, pconfig in self.config.get("providers", {}).items():
            if not pconfig.get("enabled", False):
                continue
            if not self._has_credentials(provider_name):
                continue

            provider = self._init_provider(provider_name)
            if provider is None:
                continue
            self.providers[provider_name] = provider

            for m in pconfig.get("models", []):
                self.models.append(ModelConfig(
                    id=m["id"],
                    name=m["name"],
                    provider=provider_name,
                    role=ModelRole(m.get("role", "reasoning")),
                ))

    def _has_credentials(self, provider: str) -> bool:
        key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "ollama": None,  # no key needed
        }
        env_var = key_map.get(provider)
        if env_var is None:
            return True
        # Check user-provided keys first, then env vars
        if self.api_keys.get(env_var):
            return True
        return bool(os.environ.get(env_var))

    def _init_provider(self, name: str) -> BaseProvider | None:
        key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
        api_key = self.api_keys.get(key_map.get(name, ""))
        try:
            if name == "anthropic":
                from core.providers.anthropic_provider import AnthropicProvider
                return AnthropicProvider(api_key=api_key)
            elif name == "openai":
                from core.providers.openai_provider import OpenAIProvider
                return OpenAIProvider(api_key=api_key)
            elif name == "google":
                from core.providers.google_provider import GoogleProvider
                return GoogleProvider(api_key=api_key)
            elif name == "deepseek":
                from core.providers.deepseek_provider import DeepSeekProvider
                return DeepSeekProvider(api_key=api_key)
            elif name == "ollama":
                from core.providers.ollama_provider import OllamaProvider
                return OllamaProvider()
        except Exception as e:
            print(f"[WARN] Could not init provider {name}: {e}")
            return None

    def get_active_models(self, roles: list[ModelRole] | None = None) -> list[ModelConfig]:
        if roles is None:
            return self.models
        return [m for m in self.models if m.role in roles]

    async def query_model(
        self,
        model: ModelConfig,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        provider = self.providers.get(model.provider)
        if not provider:
            return ModelResponse(
                model_id=model.id,
                model_name=model.name,
                provider=model.provider,
                content="",
                error=f"Provider {model.provider} not available",
            )

        timeout = self.config.get("pipeline", {}).get("timeout", 120)
        try:
            return await asyncio.wait_for(
                provider.timed_generate(
                    model_id=model.id,
                    model_name=model.name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return ModelResponse(
                model_id=model.id,
                model_name=model.name,
                provider=model.provider,
                content="",
                error=f"Timed out after {timeout}s",
            )

    async def query_all(
        self,
        prompt: str,
        system_prompt: str | None = None,
        models: list[ModelConfig] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> list[ModelResponse]:
        """Send prompt to all active models concurrently."""
        if models is None:
            models = self.models

        max_conc = self.config.get("pipeline", {}).get("max_concurrency", 5)
        sem = asyncio.Semaphore(max_conc)

        async def bounded(m: ModelConfig) -> ModelResponse:
            async with sem:
                return await self.query_model(
                    m, prompt, system_prompt, temperature, max_tokens
                )

        tasks = [bounded(m) for m in models]
        return await asyncio.gather(*tasks)
