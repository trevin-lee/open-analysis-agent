"""Tests for visualization tools."""

from __future__ import annotations

import numpy as np

from openanalysisagent.tools.visualization import (
    plot_brazil_band,
    plot_distribution,
    plot_pull,
)


def test_plot_distribution(tmp_dir):
    rng = np.random.default_rng(42)
    data = rng.normal(91.2, 2.5, 1000)
    contents, bin_edges = np.histogram(data, bins=30, range=(80, 100))

    output = str(tmp_dir / "dist.png")
    result = plot_distribution.invoke(
        {
            "histograms": [
                {
                    "bin_edges": bin_edges.tolist(),
                    "contents": contents.tolist(),
                    "label": "Z → μμ",
                    "histtype": "step",
                }
            ],
            "xlabel": "m_{μμ} [GeV]",
            "ylabel": "Events / 0.67 GeV",
            "cms_label": "Private Work",
            "output_path": output,
        }
    )
    assert "saved" in result.lower()
    assert (tmp_dir / "dist.png").exists()


def test_plot_distribution_with_ratio(tmp_dir):
    edges = np.linspace(0, 100, 11).tolist()
    rng = np.random.default_rng(42)

    output = str(tmp_dir / "ratio.png")
    result = plot_distribution.invoke(
        {
            "histograms": [
                {
                    "bin_edges": edges,
                    "contents": rng.poisson(100, 10).tolist(),
                    "label": "Data",
                    "histtype": "errorbar",
                },
                {
                    "bin_edges": edges,
                    "contents": [100] * 10,
                    "label": "MC",
                    "histtype": "fill",
                },
            ],
            "xlabel": "p_T [GeV]",
            "ratio_panel": True,
            "output_path": output,
        }
    )
    assert "saved" in result.lower()


def test_plot_brazil_band(tmp_dir):
    poi = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    obs = [0.9, 0.5, 0.1, 0.03, 0.005, 0.001, 0.0001]
    exp = [0.8, 0.4, 0.08, 0.02, 0.003, 0.0005, 0.00005]
    exp_1d = [0.7, 0.3, 0.05, 0.01, 0.001, 0.0001, 0.00001]
    exp_1u = [0.9, 0.55, 0.15, 0.05, 0.01, 0.003, 0.0005]
    exp_2d = [0.6, 0.2, 0.02, 0.003, 0.0003, 0.00002, 0.000001]
    exp_2u = [0.95, 0.65, 0.25, 0.1, 0.03, 0.01, 0.003]

    output = str(tmp_dir / "brazil.png")
    result = plot_brazil_band.invoke(
        {
            "poi_values": poi,
            "observed": obs,
            "expected": exp,
            "expected_1sigma": [exp_1d, exp_1u],
            "expected_2sigma": [exp_2d, exp_2u],
            "output_path": output,
        }
    )
    assert "saved" in result.lower()
    assert (tmp_dir / "brazil.png").exists()


def test_plot_pull(tmp_dir):
    output = str(tmp_dir / "pulls.png")
    result = plot_pull.invoke(
        {
            "parameter_names": ["bkg_norm", "JES", "btag_SF", "lumi"],
            "pulls": [0.3, -1.2, 0.1, 0.8],
            "constraints": [0.9, 0.85, 0.95, 0.92],
            "output_path": output,
        }
    )
    assert "saved" in result.lower()
    assert "Largest pull" in result
