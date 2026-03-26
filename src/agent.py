"""Core agent construction using LangGraph with CMS tool suite."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from src.config import OAAConfig, load_config
from src.environment import (
    detect_environment,
    get_unavailable_tools_message,
)
from src.system_prompt import build_system_prompt
from src.tools import get_all_tools


def create_cms_agent(
    model: str | None = None,
    tools: list[BaseTool] | None = None,
    system_prompt: str | None = None,
    config: OAAConfig | None = None,
    cmssw_version: str | None = None,
    **kwargs: Any,
):
    """Create a CMS analysis agent with the full tool suite.

    Args:
        model: Model string in "provider:model_name" format.
               Examples: "anthropic:claude-sonnet-4-20250514", "openai:gpt-4o"
        tools: Additional custom tools to include alongside the CMS tool suite.
        system_prompt: Override the default CMS system prompt entirely.
        config: OAAConfig instance. Loaded from file/defaults if not provided.
        cmssw_version: Override CMSSW version (sets env var).
        **kwargs: Additional arguments passed to create_react_agent.

    Returns:
        A LangGraph CompiledGraph agent ready to invoke.
    """
    import os

    if config is None:
        config = load_config()

    if cmssw_version:
        os.environ["CMSSW_VERSION"] = cmssw_version

    # Resolve model
    if model is None:
        model = config.model.model_string

    # Detect environment and get tools
    env = detect_environment()
    cms_tools = get_all_tools(env)

    # Merge custom tools
    all_tools = cms_tools
    if tools:
        all_tools = cms_tools + list(tools)

    # Build system prompt
    if system_prompt is None:
        env_notice = get_unavailable_tools_message(env)
        prompt = build_system_prompt(env_notice)
    else:
        prompt = system_prompt

    # Initialize the LLM via LangChain's model init
    from langchain.chat_models import init_chat_model

    llm = init_chat_model(model, temperature=config.model.temperature)

    # Create the agent using LangGraph's react agent
    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=SystemMessage(content=prompt),
        **kwargs,
    )

    return agent
