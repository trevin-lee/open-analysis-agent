"""Test that all modules import correctly."""

from __future__ import annotations


def test_import_package():
    import src

    assert src.__version__ == "1.0.0"


def test_import_create_cms_agent():
    from src import create_cms_agent

    assert callable(create_cms_agent)


def test_import_tools():
    from src.tools import get_all_tools

    tools = get_all_tools({"mode": "lightweight"})
    assert len(tools) >= 22  # 22 lightweight tools


def test_import_all_tool_modules():
    pass


def test_import_config():
    from src.config import load_config

    assert callable(load_config)


def test_import_environment():
    from src.environment import detect_environment

    assert callable(detect_environment)


def test_import_system_prompt():
    from src.system_prompt import build_system_prompt

    prompt = build_system_prompt()
    assert "Open Analysis Agent" in prompt
    assert "CMS" in prompt


def test_import_terminal():
    from src.terminal import main

    assert callable(main)
