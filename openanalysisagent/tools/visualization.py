"""Visualization tools — CMS-style distribution plots, Brazil bands, pull plots."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from langchain_core.tools import tool

matplotlib.use("Agg")


@tool
def plot_distribution(
    histograms: list[dict],
    xlabel: str,
    ylabel: str = "Events",
    title: str | None = None,
    cms_label: str = "Private Work",
    lumi: float | None = None,
    sqrt_s: float = 13.6,
    ratio_panel: bool = False,
    output_path: str = "distribution.png",
) -> str:
    """Create a publication-quality distribution plot with CMS style formatting.

    Each histogram dict should contain:
        - bin_edges: list of bin edge values (length = n_bins + 1)
        - contents: list of bin contents (length = n_bins)
        - label: legend label string
        - color: matplotlib color (optional)
        - histtype: "step", "fill", or "errorbar" (optional, default "step")

    Args:
        histograms: List of histogram data dicts.
        xlabel: X-axis label with units (e.g., "m_{μμ} [GeV]").
        ylabel: Y-axis label.
        title: Plot title (optional).
        cms_label: CMS label — "Preliminary", "Supplementary", "Private Work".
        lumi: Integrated luminosity in fb⁻¹ (optional).
        sqrt_s: Center-of-mass energy in TeV.
        ratio_panel: Include a data/MC ratio panel.
        output_path: Path to save the figure.

    Returns:
        Confirmation with output path.
    """
    hep.style.use("CMS")

    if ratio_panel and len(histograms) >= 2:
        fig, (ax, rax) = plt.subplots(
            2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )
        fig.subplots_adjust(hspace=0.05)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        rax = None

    default_colors = [
        "#3f90da",
        "#ffa90e",
        "#bd1f01",
        "#94a4a2",
        "#832db6",
        "#a96b59",
        "#e76300",
        "#b9ac70",
    ]

    for i, h in enumerate(histograms):
        edges = np.array(h["bin_edges"])
        contents = np.array(h["contents"])
        label = h.get("label", f"Hist {i}")
        color = h.get("color", default_colors[i % len(default_colors)])
        histtype = h.get("histtype", "step")

        if histtype == "errorbar":
            centers = (edges[:-1] + edges[1:]) / 2
            errors = np.sqrt(contents)
            ax.errorbar(
                centers, contents, yerr=errors, fmt="o", color=color, label=label, markersize=4
            )
        elif histtype == "fill":
            hep.histplot(
                contents, edges, ax=ax, label=label, color=color, histtype="fill", alpha=0.7
            )
        else:
            hep.histplot(
                contents, edges, ax=ax, label=label, color=color, histtype="step", linewidth=2
            )

    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)

    if title:
        ax.set_title(title, fontsize=14)

    # CMS label
    hep.cms.label(cms_label, data=True, lumi=lumi, year=None, ax=ax, com=sqrt_s)

    # Ratio panel
    if rax is not None and len(histograms) >= 2:
        num = np.array(histograms[0]["contents"])
        den = np.array(histograms[1]["contents"])
        edges = np.array(histograms[0]["bin_edges"])

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(den > 0, num / den, 0)
            ratio_err = np.where(den > 0, np.sqrt(num) / den, 0)

        centers = (edges[:-1] + edges[1:]) / 2
        rax.errorbar(centers, ratio, yerr=ratio_err, fmt="o", color="black", markersize=4)
        rax.axhline(1, color="gray", linestyle="--", linewidth=1)
        rax.set_ylabel("Ratio")
        rax.set_ylim(0.5, 1.5)
        rax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(xlabel)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return f"Plot saved to: {output_path}"


@tool
def plot_brazil_band(
    poi_values: list[float],
    observed: list[float],
    expected: list[float],
    expected_1sigma: list[list[float]],
    expected_2sigma: list[list[float]],
    xlabel: str = "Signal strength μ",
    output_path: str = "brazil_band.png",
    cms_label: str = "Private Work",
    lumi: float | None = None,
    sqrt_s: float = 13.6,
) -> str:
    """Create a CLs limit plot with expected and observed limits (Brazil band).

    Args:
        poi_values: Scanned parameter of interest values.
        observed: Observed CLs values at each scan point.
        expected: Expected (median) CLs values.
        expected_1sigma: [down_1sigma, up_1sigma] expected bands (lists of values).
        expected_2sigma: [down_2sigma, up_2sigma] expected bands (lists of values).
        xlabel: X-axis label.
        output_path: Path to save the figure.
        cms_label: CMS label text.
        lumi: Integrated luminosity in fb⁻¹.
        sqrt_s: Center-of-mass energy in TeV.

    Returns:
        Confirmation with output path and observed/expected limit values.
    """
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8))

    x = np.array(poi_values)
    obs = np.array(observed)
    exp = np.array(expected)
    exp_1down = np.array(expected_1sigma[0])
    exp_1up = np.array(expected_1sigma[1])
    exp_2down = np.array(expected_2sigma[0])
    exp_2up = np.array(expected_2sigma[1])

    # 2σ band (yellow)
    ax.fill_between(x, exp_2down, exp_2up, color="#ffff00", label="Expected ±2σ")
    # 1σ band (green)
    ax.fill_between(x, exp_1down, exp_1up, color="#00ff00", label="Expected ±1σ")
    # Expected (dashed)
    ax.plot(x, exp, "k--", linewidth=2, label="Expected")
    # Observed (solid)
    ax.plot(x, obs, "k-", linewidth=2, marker="o", markersize=5, label="Observed")
    # CL=0.05 line
    ax.axhline(0.05, color="red", linestyle=":", linewidth=1, label="95% CL")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CL_s")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1)
    ax.legend(frameon=False, loc="upper right")

    hep.cms.label(cms_label, data=True, lumi=lumi, ax=ax, com=sqrt_s)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Find crossing points
    obs_limit = _find_crossing(x, obs, 0.05)
    exp_limit = _find_crossing(x, exp, 0.05)

    result = [
        f"Brazil band plot saved to: {output_path}",
        f"  Observed 95% CL limit: μ < {obs_limit:.4f}"
        if obs_limit
        else "  Observed limit: not found in scan range",
        f"  Expected 95% CL limit: μ < {exp_limit:.4f}"
        if exp_limit
        else "  Expected limit: not found in scan range",
    ]

    return "\n".join(result)


def _find_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    """Find where y crosses below threshold via linear interpolation."""
    for i in range(len(y) - 1):
        if (y[i] >= threshold) != (y[i + 1] >= threshold):
            # Linear interpolation
            frac = (threshold - y[i]) / (y[i + 1] - y[i])
            return float(x[i] + frac * (x[i + 1] - x[i]))
    return None


@tool
def plot_pull(
    parameter_names: list[str],
    pulls: list[float],
    constraints: list[float],
    output_path: str = "pulls.png",
    cms_label: str = "Private Work",
) -> str:
    """Create nuisance parameter pull plots from fit results.

    Args:
        parameter_names: Names of nuisance parameters.
        pulls: Pull values (best-fit - nominal) / σ.
        constraints: Post-fit constraint values (σ_post / σ_pre).
        output_path: Path to save the figure.
        cms_label: CMS label text.

    Returns:
        Confirmation with output path.
    """
    hep.style.use("CMS")

    n = len(parameter_names)
    if n == 0:
        return "No parameters to plot."

    fig_height = max(4, 0.4 * n + 2)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    y_pos = np.arange(n)
    pulls_arr = np.array(pulls)
    constraints_arr = np.array(constraints)

    # ±2σ and ±1σ bands
    ax.axvspan(-2, 2, color="#ffff00", alpha=0.3)
    ax.axvspan(-1, 1, color="#00ff00", alpha=0.3)
    ax.axvline(0, color="gray", linewidth=1)

    # Pull values
    ax.errorbar(
        pulls_arr, y_pos, xerr=constraints_arr, fmt="o", color="black", markersize=5, capsize=3
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(parameter_names, fontsize=8)
    ax.set_xlabel("(θ̂ - θ₀) / Δθ")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, n - 0.5)

    ax.set_title(f"{cms_label}", fontsize=12, loc="left")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary stats
    max_pull_idx = int(np.argmax(np.abs(pulls_arr)))
    lines = [
        f"Pull plot saved to: {output_path}",
        f"  Parameters: {n}",
        f"  Largest pull: {parameter_names[max_pull_idx]} ({pulls_arr[max_pull_idx]:+.2f}σ)",
        f"  Pulls > 1σ: {int(np.sum(np.abs(pulls_arr) > 1))}",
        f"  Pulls > 2σ: {int(np.sum(np.abs(pulls_arr) > 2))}",
    ]

    return "\n".join(lines)
