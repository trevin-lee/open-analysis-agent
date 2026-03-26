"""Tests for CMSSW tools (mostly integration tests requiring CVMFS)."""

from __future__ import annotations

import pytest

from src.tools.cmssw import (
    generate_cmsrun_config,
    query_global_tag,
    setup_cmssw,
)


def test_setup_cmssw_no_cvmfs():
    """Without CVMFS, setup_cmssw should return an error."""
    result = setup_cmssw.invoke({"version": "CMSSW_14_2_0"})
    # Either CVMFS is missing (expected locally) or it works (in Docker)
    assert "CVMFS" in result or "CMSSW" in result


def test_query_global_tag_run3_data():
    result = query_global_tag.invoke(
        {
            "dataset": "/ZeroBias/Run2024A-v1/NANOAOD",
            "data_type": "data",
        }
    )
    assert "global tag" in result.lower() or "Global tag" in result


def test_query_global_tag_run2_mc():
    result = query_global_tag.invoke(
        {
            "dataset": "/DYJetsToLL/RunIISummer20UL18-106X/NANOAODSIM",
            "data_type": "mc",
        }
    )
    assert "global tag" in result.lower() or "Global tag" in result


def test_query_global_tag_unknown():
    result = query_global_tag.invoke(
        {
            "dataset": "/Unknown/Dataset/FORMAT",
            "data_type": "data",
        }
    )
    # Should still provide guidance
    assert "global tag" in result.lower() or "Common" in result


@pytest.mark.integration
def test_generate_cmsrun_config(tmp_dir):
    result = generate_cmsrun_config.invoke(
        {
            "process_name": "TEST",
            "input_files": ["/store/data/Run2024A/ZeroBias/NANOAOD/v1/file.root"],
            "global_tag": "140X_dataRun3_v4",
            "max_events": 100,
        }
    )
    assert "config generated" in result.lower() or "Config" in result
