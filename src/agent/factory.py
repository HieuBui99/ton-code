from __future__ import annotations

import logging
from typing import Literal

from openai import AsyncOpenAI
from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.agent import AgentRetries
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from src.agent.deps import AgentDeps
from src.config.settings import settings
from src.tools.registry import register_tools

logger = logging.getLogger(__name__)

_BASE_INSTRUCTIONS = (
    "You are TonCode, a terminal coding agent that helps a developer in their working "
    "directory. You are concise and precise: answer directly, prefer running the work over "
    "describing it, and never invent file contents or command output you have not seen. "
    "When you do not have a tool for something yet, say so plainly rather than pretending."
)


def build_agent(*, model: str | None = None) -> Agent[AgentDeps, str | DeferredToolRequests]:
    """
    Construct the agent on the configured LLM Provider + register the tools
    """
    built_model = _build_model(model=model)
    agent: Agent[AgentDeps, str | DeferredToolRequests] = Agent(
        built_model,
        deps_type=AgentDeps,
        output_type=[str, DeferredToolRequests],
        retries=AgentRetries(output=3, tools=5),
    )
    register_tools(agent)
    _register_instructions(agent)

    return agent


def _build_model(*, model: str | None = None) -> Model:
    provider = settings.llm_provider
    if provider == "gemini":
        return GoogleModel(
            model or settings.gemini_model,
            provider=GoogleProvider(api_key=_provider_api_key("gemini")),
        )
    elif provider == "openai":
        base_url = f"{settings.model_endpoint_url}/v1"
        api_key = _provider_api_key("openai")
        if api_key:
            client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
            )
        else:
            client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
        return OpenAIChatModel(
            model or settings.model_endpoint_model,
            provider=OpenAIProvider(openai_client=client),
        )
    raise ValueError(f"unsupported llm_provider: {provider!r}")


def _provider_api_key(provider: Literal["gemini", "openrouter"]) -> str:
    secret = settings.gemini_api_key if provider == "gemini" else settings.model_endpoint_api_key
    return secret.get_secret_value()


def _register_instructions(agent: Agent[AgentDeps, str | DeferredToolRequests]) -> None:
    @agent.instructions
    def assemble_instructions(ctx: RunContext[AgentDeps]) -> str:
        harness_home = ctx.deps.harness_home or ctx.deps.cwd
        parts = (
            _BASE_INSTRUCTIONS,
            # Add memory, skills, ...
        )
        return "\n\n".join(part for part in parts if part)
