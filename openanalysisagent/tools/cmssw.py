"""CMSSW tools — environment setup, cmsRun, config generation, global tags.

These tools require the Docker + CVMFS environment.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from langchain_core.tools import tool


def _run_cms_command(cmd: str, env: dict | None = None, timeout: int = 300) -> tuple[int, str, str]:
    """Run a shell command with CMS environment sourced."""
    full_cmd = "source /cvmfs/cms.cern.ch/cmsset_default.sh 2>/dev/null; " + cmd
    result = subprocess.run(
        full_cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env or os.environ.copy(),
    )
    return result.returncode, result.stdout, result.stderr


@tool
def setup_cmssw(
    version: str = "CMSSW_14_2_0",
    scram_arch: str | None = None,
    workdir: str | None = None,
) -> str:
    """Set up a CMSSW working environment.

    Creates a CMSSW area, initializes the environment, and validates.
    Requires Docker + CVMFS environment.

    Args:
        version: CMSSW version (e.g., "CMSSW_14_2_0").
        scram_arch: Architecture string. Auto-detected if not provided.
        workdir: Working directory path. Defaults to /workspace.

    Returns:
        Confirmation with CMSSW environment details.
    """
    if not Path("/cvmfs/cms.cern.ch").exists():
        return (
            "Error: CVMFS is not mounted. CMSSW tools require the Docker + CVMFS environment.\n"
            "Run: docker compose up"
        )

    if workdir is None:
        workdir = "/workspace"

    if scram_arch is None:
        scram_arch = os.environ.get("SCRAM_ARCH", "el9_amd64_gcc12")

    cmssw_path = Path(workdir) / version

    if cmssw_path.exists():
        return (
            f"CMSSW area already exists: {cmssw_path}\n"
            f"  Version: {version}\n"
            f"  SCRAM_ARCH: {scram_arch}\n"
            f"  Source: {cmssw_path / 'src'}"
        )

    cmd = (
        f"export SCRAM_ARCH={scram_arch}; "
        f"cd {workdir}; "
        f"cmsrel {version}; "
        f"cd {version}/src; "
        f"cmsenv; "
        f"echo CMSSW_BASE=$CMSSW_BASE; "
        f"echo SCRAM_ARCH=$SCRAM_ARCH"
    )

    rc, stdout, stderr = _run_cms_command(cmd, timeout=120)

    if rc != 0:
        return f"Error setting up {version}:\n{stderr}\n{stdout}"

    return (
        f"CMSSW environment set up successfully.\n"
        f"  Version: {version}\n"
        f"  SCRAM_ARCH: {scram_arch}\n"
        f"  Path: {cmssw_path}\n"
        f"  Source area: {cmssw_path / 'src'}\n"
        f"\n{stdout}"
    )


@tool
def run_cmsrun(
    config_path: str,
    num_events: int | None = None,
    num_threads: int | None = None,
) -> str:
    """Execute a cmsRun job with the given configuration file.

    Requires Docker + CVMFS environment with CMSSW set up.

    Args:
        config_path: Path to the CMSSW Python configuration file.
        num_events: Number of events to process (-1 for all).
        num_threads: Number of threads.

    Returns:
        Job summary with exit code, output files, and processing statistics.
    """
    if not Path("/cvmfs/cms.cern.ch").exists():
        return "Error: CVMFS is not mounted. Use Docker + CVMFS environment."

    config_file = Path(config_path)
    if not config_file.exists():
        return f"Config file not found: {config_path}"

    # Build cmsRun command
    cmd = f"cmsRun {config_path}"
    if num_events is not None:
        cmd += f" maxEvents={num_events}"
    if num_threads is not None:
        cmd += f" numThreads={num_threads}"

    # Detect CMSSW environment from config path
    cmssw_base = os.environ.get("CMSSW_BASE", "")
    if cmssw_base:
        cmd = f"cd {cmssw_base}/src && cmsenv && {cmd}"

    try:
        rc, stdout, stderr = _run_cms_command(cmd, timeout=600)
    except subprocess.TimeoutExpired:
        return "Error: cmsRun timed out after 10 minutes."

    # Parse output for key information
    lines = [
        "cmsRun completed.",
        f"  Config: {config_path}",
        f"  Exit code: {rc}",
    ]

    if num_events is not None:
        lines.append(f"  Max events: {num_events}")

    # Look for output file info in stdout
    for line in stdout.split("\n"):
        if "TrigReport" in line and "Events" in line:
            lines.append(f"  {line.strip()}")
        elif "output" in line.lower() and ".root" in line.lower():
            lines.append(f"  Output: {line.strip()}")

    if rc != 0:
        lines.append("\nStderr (last 20 lines):")
        stderr_lines = stderr.strip().split("\n")
        for line in stderr_lines[-20:]:
            lines.append(f"  {line}")

    return "\n".join(lines)


@tool
def generate_cmsrun_config(
    process_name: str,
    input_files: list[str],
    global_tag: str,
    output_file: str | None = None,
    max_events: int = -1,
) -> str:
    """Generate a CMSSW Python configuration file from high-level parameters.

    Args:
        process_name: Process name (e.g., "ANALYSIS").
        input_files: Input file paths or DAS dataset names.
        global_tag: Conditions global tag.
        output_file: Output ROOT file path.
        max_events: Maximum events to process.

    Returns:
        Generated configuration file path and contents summary.
    """
    # Format input files
    input_strs = []
    for f in input_files:
        if f.startswith("/store/"):
            input_strs.append(f"'root://cms-xrd-global.cern.ch/{f}'")
        else:
            input_strs.append(f"'{f}'")

    input_list = ",\n        ".join(input_strs)

    output_section = ""
    if output_file:
        output_section = f"""
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('{output_file}')
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('{output_file}'),
    outputCommands = cms.untracked.vstring('keep *')
)
process.output_step = cms.EndPath(process.output)
"""

    config = f"""import FWCore.FrameworkCore.FWCore as cms
import FWCore.ParameterSet.Config as cms

process = cms.Process("{process_name}")

# Message logger
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

# Input
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32({max_events}))
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        {input_list}
    )
)

# Conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '{global_tag}', '')

# Geometry and magnetic field
process.load("Configuration.Geometry.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
{output_section}
# Schedule
process.p = cms.Path()
"""

    # Write config file
    config_path = f"/workspace/{process_name.lower()}_cfg.py"
    try:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            f.write(config)
    except Exception as e:
        return f"Error writing config file: {e}"

    return (
        f"CMSSW config generated: {config_path}\n"
        f"  Process: {process_name}\n"
        f"  Input files: {len(input_files)}\n"
        f"  Global tag: {global_tag}\n"
        f"  Max events: {max_events}\n"
        f"  Output: {output_file or 'none'}\n"
        f"\nRun with: cmsRun {config_path}"
    )


@tool
def query_global_tag(
    dataset: str,
    data_type: str = "data",
    era: str | None = None,
) -> str:
    """Recommend the correct global tag for a given dataset and era.

    Args:
        dataset: Dataset name or DAS path.
        data_type: "data" or "mc".
        era: Data-taking era (e.g., "Run2024A", "Run2023D").

    Returns:
        Recommended global tag with explanation.
    """
    # Global tag recommendations database
    # These are commonly used global tags — the agent should verify with TWiki
    gt_map = {
        "data": {
            "Run2024": "140X_dataRun3_v4",
            "Run2023": "130X_dataRun3_v2",
            "Run2022": "124X_dataRun3_v15",
            "Run2018": "106X_dataRun2_v37",
            "Run2017": "106X_dataRun2_v37",
            "Run2016": "106X_dataRun2_v37",
        },
        "mc": {
            "Run3Summer23": "130X_mcRun3_2023_realistic_v15",
            "Run3Summer22": "126X_mcRun3_2022_realistic_v2",
            "RunIISummer20UL18": "106X_upgrade2018_realistic_v16_L1v1",
            "RunIISummer20UL17": "106X_mc2017_realistic_v9",
            "RunIISummer20UL16": "106X_mcRun2_asymptotic_v17",
        },
    }

    # Try to match era
    detected_era = era
    if not detected_era and dataset:
        for key in list(gt_map.get(data_type, {}).keys()):
            if key in dataset:
                detected_era = key
                break

    if not detected_era:
        # Provide general guidance
        lines = [
            f"Could not auto-detect era for dataset: {dataset}",
            f"Data type: {data_type}",
            "",
            "Common global tags:",
        ]
        for era_key, gt in gt_map.get(data_type, {}).items():
            lines.append(f"  {era_key}: {gt}")
        lines.append(
            "\nVerify the correct global tag at:\n"
            "  https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis"
        )
        return "\n".join(lines)

    gt = gt_map.get(data_type, {}).get(detected_era)
    if not gt:
        # Partial match
        for key, val in gt_map.get(data_type, {}).items():
            if detected_era.startswith(key[:4]):
                gt = val
                break

    if gt:
        return (
            f"Recommended global tag:\n"
            f"  Dataset: {dataset}\n"
            f"  Type: {data_type}\n"
            f"  Era: {detected_era}\n"
            f"  Global tag: {gt}\n"
            f"\nVerify at: https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis"
        )

    return (
        f"No global tag found for era '{detected_era}' ({data_type}).\n"
        f"Check https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis"
    )
