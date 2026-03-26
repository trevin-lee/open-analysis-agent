"""Configuration loading and defaults for OpenAnalysisAgent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    provider: str = "anthropic"
    name: str = "claude-sonnet-4-20250514"
    temperature: float = 0

    @property
    def model_string(self) -> str:
        return f"{self.provider}:{self.name}"


@dataclass
class EnvironmentConfig:
    cmssw_version: str = "CMSSW_14_2_0"
    scram_arch: str = "el9_amd64_gcc12"
    default_data_dir: str = "/workspace/data"
    default_output_dir: str = "/workspace/output"


@dataclass
class AnalysisConfig:
    default_cms_label: str = "Private Work"
    default_sqrt_s: float = 13.6
    default_lumi: float | None = None
    blinding: bool = True


@dataclass
class AgentConfig:
    max_iterations: int = 50
    human_approval_steps: list[str] = field(
        default_factory=lambda: ["cut_definition", "fit_configuration", "unblinding"]
    )
    log_level: str = "INFO"
    artifact_dir: str = "/workspace/artifacts"


@dataclass
class OAAConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


def load_config(path: str | Path | None = None) -> OAAConfig:
    """Load configuration from YAML file, falling back to defaults.

    Search order:
    1. Explicit path argument
    2. OAA_CONFIG env var
    3. ./openanalysisagent.yml
    4. Defaults
    """
    if path is None:
        path = os.environ.get("OAA_CONFIG", "openanalysisagent.yml")

    config = OAAConfig()
    config_path = Path(path)

    if not config_path.exists():
        # Apply env var overrides
        model_override = os.environ.get("OAA_MODEL")
        if model_override and ":" in model_override:
            provider, name = model_override.split(":", 1)
            config.model.provider = provider
            config.model.name = name
        return config

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    if "model" in raw:
        m = raw["model"]
        config.model = ModelConfig(
            provider=m.get("provider", config.model.provider),
            name=m.get("name", config.model.name),
            temperature=m.get("temperature", config.model.temperature),
        )

    if "environment" in raw:
        e = raw["environment"]
        config.environment = EnvironmentConfig(
            cmssw_version=e.get("cmssw_version", config.environment.cmssw_version),
            scram_arch=e.get("scram_arch", config.environment.scram_arch),
            default_data_dir=e.get("default_data_dir", config.environment.default_data_dir),
            default_output_dir=e.get("default_output_dir", config.environment.default_output_dir),
        )

    if "analysis" in raw:
        a = raw["analysis"]
        config.analysis = AnalysisConfig(
            default_cms_label=a.get("default_cms_label", config.analysis.default_cms_label),
            default_sqrt_s=a.get("default_sqrt_s", config.analysis.default_sqrt_s),
            default_lumi=a.get("default_lumi", config.analysis.default_lumi),
            blinding=a.get("blinding", config.analysis.blinding),
        )

    if "agent" in raw:
        ag = raw["agent"]
        config.agent = AgentConfig(
            max_iterations=ag.get("max_iterations", config.agent.max_iterations),
            human_approval_steps=ag.get("human_approval_steps", config.agent.human_approval_steps),
            log_level=ag.get("log_level", config.agent.log_level),
            artifact_dir=ag.get("artifact_dir", config.agent.artifact_dir),
        )

    # Env var overrides always win
    model_override = os.environ.get("OAA_MODEL")
    if model_override and ":" in model_override:
        provider, name = model_override.split(":", 1)
        config.model.provider = provider
        config.model.name = name

    return config
