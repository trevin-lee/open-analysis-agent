"""LangGraph API server for Open Analysis Agent."""

from __future__ import annotations

from src.agent import create_cms_agent


def create_graph():
    """Create the agent graph for LangGraph serve."""
    return create_cms_agent()


if __name__ == "__main__":
    from src.terminal import main

    main()
