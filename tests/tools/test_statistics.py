"""Tests for statistical inference tools."""

from __future__ import annotations

from src.tools.statistics import (
    build_pyhf_model,
    run_cls_limit,
    run_discovery_test,
    run_fit,
)


def test_build_pyhf_model():
    result = build_pyhf_model.invoke(
        {
            "signal": [5.0, 10.0, 5.0],
            "background": [50.0, 100.0, 50.0],
            "observed": [48.0, 105.0, 52.0],
        }
    )
    assert "workspace built successfully" in result
    assert "Signal yield" in result


def test_build_pyhf_model_with_uncertainties():
    result = build_pyhf_model.invoke(
        {
            "signal": [5.0, 10.0, 5.0],
            "background": [50.0, 100.0, 50.0],
            "observed": [48.0, 105.0, 52.0],
            "signal_uncertainty": [1.0, 2.0, 1.0],
            "background_uncertainty": [5.0, 10.0, 5.0],
        }
    )
    assert "workspace built successfully" in result
    assert "signal_syst" in result
    assert "bkg_syst" in result


def test_build_pyhf_model_mismatched_bins():
    result = build_pyhf_model.invoke(
        {
            "signal": [5.0, 10.0],
            "background": [50.0, 100.0, 50.0],
            "observed": [48.0, 105.0, 52.0],
        }
    )
    assert "Error" in result or "same number of bins" in result


def test_run_fit(sample_workspace):
    result = run_fit.invoke({"workspace": sample_workspace})
    assert "Maximum Likelihood Fit" in result
    assert "POI" in result


def test_run_discovery_test(sample_workspace):
    result = run_discovery_test.invoke({"workspace": sample_workspace})
    assert "Discovery Test" in result
    assert "significance" in result.lower() or "σ" in result


def test_run_cls_limit(sample_workspace):
    result = run_cls_limit.invoke(
        {
            "workspace": sample_workspace,
            "poi_values": [0.0, 0.5, 1.0, 2.0, 3.0, 5.0],
        }
    )
    assert "CLs" in result or "limit" in result.lower()
