"""Tests for analysis tools."""

from __future__ import annotations

import numpy as np

from src.tools.analysis import (
    apply_event_selection,
    compute_delta_r,
    compute_invariant_mass,
    fill_histogram,
)


def test_apply_event_selection(sample_root_file):
    result = apply_event_selection.invoke(
        {
            "filepath": str(sample_root_file),
            "tree_name": "Events",
            "cuts": {"nMuon": [">=", 2]},
        }
    )
    assert "Events before cuts" in result
    assert "Events after cuts" in result
    assert "efficiency" in result.lower() or "Eff" in result


def test_compute_invariant_mass():
    # Z boson-like dimuon mass: two muons near 91 GeV
    rng = np.random.default_rng(42)
    n = 100
    result = compute_invariant_mass.invoke(
        {
            "pt": [rng.uniform(20, 60, n).tolist(), rng.uniform(20, 60, n).tolist()],
            "eta": [rng.uniform(-2.4, 2.4, n).tolist(), rng.uniform(-2.4, 2.4, n).tolist()],
            "phi": [rng.uniform(-np.pi, np.pi, n).tolist(), rng.uniform(-np.pi, np.pi, n).tolist()],
            "mass": [[0.1057] * n, [0.1057] * n],
        }
    )
    assert "Invariant mass" in result
    assert "Mean" in result


def test_compute_invariant_mass_too_few():
    result = compute_invariant_mass.invoke(
        {
            "pt": [[1.0, 2.0]],
            "eta": [[0.0, 0.0]],
            "phi": [[0.0, 0.0]],
            "mass": [[0.0, 0.0]],
        }
    )
    assert "Need at least 2" in result


def test_compute_delta_r():
    result = compute_delta_r.invoke(
        {
            "eta1": [0.0, 1.0, -1.0],
            "phi1": [0.0, 0.5, -0.5],
            "eta2": [0.5, 1.5, -0.5],
            "phi2": [0.5, 1.0, 0.0],
        }
    )
    assert "ΔR" in result
    assert "Mean" in result


def test_fill_histogram():
    rng = np.random.default_rng(42)
    data = rng.normal(91.2, 2.5, 500).tolist()
    result = fill_histogram.invoke(
        {
            "data": data,
            "bins": 30,
            "hist_range": [80.0, 100.0],
            "label": "Z mass",
        }
    )
    assert "Histogram" in result
    assert "Mean" in result
    assert "Integral" in result


def test_fill_histogram_with_weights():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    weights = [2.0, 1.0, 1.0, 1.0, 0.5]
    result = fill_histogram.invoke(
        {
            "data": data,
            "bins": 5,
            "weights": weights,
        }
    )
    assert "Histogram" in result
