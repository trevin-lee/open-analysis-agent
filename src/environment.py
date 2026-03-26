"""Runtime environment detection for CMS software stack availability."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def detect_environment() -> dict:
    """Detect available CMS software components and return capabilities map.

    Returns a dict with keys:
        cvmfs (bool): Whether CVMFS is mounted
        cmssw (str | None): Path to CMS software on CVMFS
        cmssw_version (str | None): CMSSW version from env or default
        scram_arch (str | None): SCRAM architecture string
        madgraph (str | None): Path to MadGraph installation
        pythia (bool): Whether Pythia is available
        root (str | None): Path to ROOT installation
        delphes (str | None): Path to Delphes installation
        mode (str): "full" if CVMFS available, else "lightweight"
    """
    cvmfs_cms = Path("/cvmfs/cms.cern.ch")
    cvmfs_sft = Path("/cvmfs/sft.cern.ch")
    has_cvmfs = cvmfs_cms.exists() and cvmfs_sft.exists()

    cmssw_version = os.environ.get("CMSSW_VERSION", "CMSSW_14_2_0")
    scram_arch = os.environ.get("SCRAM_ARCH", "el9_amd64_gcc12")

    env: dict = {
        "cvmfs": has_cvmfs,
        "cmssw": str(cvmfs_cms) if has_cvmfs else None,
        "cmssw_version": cmssw_version if has_cvmfs else None,
        "scram_arch": scram_arch if has_cvmfs else None,
        "madgraph": None,
        "pythia": False,
        "root": None,
        "delphes": None,
        "mode": "full" if has_cvmfs else "lightweight",
    }

    if has_cvmfs:
        # Locate MadGraph
        mg_search = cvmfs_cms / scram_arch / "cms" / "madgraph"
        if mg_search.exists():
            mg_dirs = sorted(mg_search.iterdir(), reverse=True)
            if mg_dirs:
                env["madgraph"] = str(mg_dirs[0])

        # Check for ROOT
        root_path = cvmfs_sft / "lcg" / "app" / "releases" / "ROOT"
        if root_path.exists():
            env["root"] = str(root_path)

        env["pythia"] = True

    # Also check if ROOT is available as a system command
    if env["root"] is None and shutil.which("root"):
        env["root"] = shutil.which("root")

    return env


def get_available_tool_categories(env: dict | None = None) -> list[str]:
    """Return which tool categories are available in the current environment."""
    if env is None:
        env = detect_environment()

    categories = ["data_io", "literature", "analysis", "statistics", "visualization"]

    if env["mode"] == "full":
        categories.append("cmssw")

    return categories


def get_unavailable_tools_message(env: dict | None = None) -> str:
    """Return a message about unavailable tools for the current environment."""
    if env is None:
        env = detect_environment()

    if env["mode"] == "full":
        return ""

    return (
        "Running in lightweight mode (no CVMFS). "
        "CMSSW tools (setup_cmssw, run_cmsrun, generate_cmsrun_config, query_global_tag) "
        "and simulation tools (MadGraph, Pythia, Delphes) are unavailable. "
        "Use the Docker + CVMFS environment for the full software stack: "
        "docker compose up"
    )
