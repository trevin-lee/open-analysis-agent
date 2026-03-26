"""Tests for configuration and environment detection."""

from __future__ import annotations

import tempfile

from src.config import OAAConfig, load_config
from src.environment import detect_environment, get_available_tool_categories


def test_default_config():
    config = OAAConfig()
    assert config.model.provider == "anthropic"
    assert config.model.temperature == 0
    assert config.analysis.blinding is True
    assert config.agent.max_iterations == 50


def test_load_config_no_file():
    config = load_config("/nonexistent/path.yml")
    assert isinstance(config, OAAConfig)
    assert config.model.provider == "anthropic"


def test_load_config_from_yaml():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write("""
model:
  provider: openai
  name: gpt-4o
  temperature: 0.1
analysis:
  default_cms_label: "Preliminary"
  blinding: false
""")
        f.flush()
        config = load_config(f.name)

    assert config.model.provider == "openai"
    assert config.model.name == "gpt-4o"
    assert config.model.temperature == 0.1
    assert config.analysis.default_cms_label == "Preliminary"
    assert config.analysis.blinding is False


def test_model_string():
    config = OAAConfig()
    assert config.model.model_string == "anthropic:claude-sonnet-4-20250514"


def test_detect_environment():
    env = detect_environment()
    assert isinstance(env, dict)
    assert "mode" in env
    assert "cvmfs" in env
    assert env["mode"] in ("full", "lightweight")


def test_get_available_tool_categories():
    # In local environment (no CVMFS), should not include cmssw
    env = {"mode": "lightweight"}
    cats = get_available_tool_categories(env)
    assert "data_io" in cats
    assert "literature" in cats
    assert "cmssw" not in cats

    # In full environment, should include cmssw
    env_full = {"mode": "full"}
    cats_full = get_available_tool_categories(env_full)
    assert "cmssw" in cats_full
