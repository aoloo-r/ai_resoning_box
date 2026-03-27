"""Main pipeline: ties orchestrator + synthesizer + data collector together."""

from __future__ import annotations
import time
from core.models import ModelRole, SynthesisResult, SynthesisStrategy
from core.orchestrator import Orchestrator
from core.synthesizer import Synthesizer
from core.collector import DataCollector


class EnsemblePipeline:
    """High-level API for the ensemble reasoning platform."""

    def __init__(self, config_path: str | None = None, api_keys: dict[str, str] | None = None):
        self.orchestrator = Orchestrator(config_path, api_keys=api_keys)
        self.collector = DataCollector()
        self._init_synthesizer()

    def _init_synthesizer(self):
        synth_config = self.orchestrator.config.get("synthesis", {})
        provider_name = synth_config.get("provider", "anthropic")
        model_id = synth_config.get("model", "claude-sonnet-4-6")

        provider = self.orchestrator.providers.get(provider_name)
        if not provider:
            # Fall back to first available provider
            for name, p in self.orchestrator.providers.items():
                provider = p
                provider_name = name
                if self.orchestrator.models:
                    model_id = self.orchestrator.models[0].id
                break

        if not provider:
            self.synthesizer = None
        else:
            self.synthesizer = Synthesizer(provider, model_id)

    def get_status(self) -> dict:
        """Return current platform status."""
        return {
            "active_providers": list(self.orchestrator.providers.keys()),
            "active_models": [
                {"id": m.id, "name": m.name, "provider": m.provider, "role": m.role.value}
                for m in self.orchestrator.models
            ],
            "synthesis_strategy": self.orchestrator.config.get("synthesis", {}).get("strategy", "weighted_merge"),
        }

    async def run(
        self,
        query: str,
        system_prompt: str | None = None,
        strategy: SynthesisStrategy | None = None,
        roles: list[ModelRole] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> SynthesisResult:
        """Run the full ensemble pipeline."""
        if strategy is None:
            strategy = SynthesisStrategy(
                self.orchestrator.config.get("synthesis", {}).get("strategy", "weighted_merge")
            )

        if not self.synthesizer:
            raise RuntimeError(
                "No AI providers available. Set at least one API key:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...\n"
                "  export OPENAI_API_KEY=sk-...\n"
                "  export GOOGLE_API_KEY=AIza...\n"
                "Or copy .env.example to .env and fill in your keys."
            )

        models = self.orchestrator.get_active_models(roles)
        if not models:
            raise RuntimeError("No models available for the requested roles.")

        # Default system prompt for ensemble reasoning
        if system_prompt is None:
            system_prompt = (
                "You are a world-class expert. Provide thorough, accurate, well-reasoned answers. "
                "Show your reasoning process. If uncertain about something, say so explicitly."
            )

        # Query all models in parallel
        responses = await self.orchestrator.query_all(
            prompt=query,
            system_prompt=system_prompt,
            models=models,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Synthesize
        debate_rounds = self.orchestrator.config.get("synthesis", {}).get("debate_rounds", 2)
        result = await self.synthesizer.synthesize(
            query=query,
            responses=responses,
            strategy=strategy,
            orchestrator=self.orchestrator,
            debate_rounds=debate_rounds,
        )

        # Collect training data
        try:
            self.collector.collect(result)
        except Exception:
            pass  # Don't fail queries if collection errors

        return result
