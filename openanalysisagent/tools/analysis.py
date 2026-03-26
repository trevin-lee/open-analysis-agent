"""Analysis tools — event selection, kinematics, histogramming."""

from __future__ import annotations

import awkward as ak
import numpy as np
import uproot
from langchain_core.tools import tool


@tool
def apply_event_selection(
    filepath: str,
    tree_name: str,
    cuts: dict,
) -> str:
    """Apply cuts to event data and return surviving events with efficiency.

    Each cut is specified as branch_name: [operator, value] where operator
    is one of ">", ">=", "<", "<=", "==", "!=".

    Args:
        filepath: Path to ROOT/NanoAOD file.
        tree_name: TTree name.
        cuts: Dictionary of branch_name: [operator, value] pairs.
              Example: {"Muon_pt": [">", 20.0], "nMuon": ["==", 2]}

    Returns:
        Number of events before/after cuts, efficiency, and cut flow table.
    """
    try:
        f = uproot.open(filepath)
        tree = f[tree_name]
    except Exception as e:
        return f"Error opening '{filepath}': {e}"

    branches_needed = list(cuts.keys())
    try:
        arrays = tree.arrays(branches_needed, library="ak")
    except Exception as e:
        return f"Error reading branches: {e}"

    n_total = len(arrays)
    mask = np.ones(n_total, dtype=bool)

    cut_flow = [f"{'Cut':<40} {'Pass':>10} {'Eff (%)':>10} {'Cum. Eff':>10}"]
    cut_flow.append("-" * 74)
    cut_flow.append(f"{'Initial':.<40} {n_total:>10} {'100.00':>10} {'100.00':>10}")

    ops = {
        ">": lambda a, v: a > v,
        ">=": lambda a, v: a >= v,
        "<": lambda a, v: a < v,
        "<=": lambda a, v: a <= v,
        "==": lambda a, v: a == v,
        "!=": lambda a, v: a != v,
    }

    for branch, (op, value) in cuts.items():
        if op not in ops:
            return f"Unsupported operator '{op}'. Use: >, >=, <, <=, ==, !="

        arr = arrays[branch]

        # Handle jagged arrays: if variable-length (e.g., Muon_pt),
        # apply cut per-element and use ak.any/ak.all
        try:
            if arr.ndim > 1:
                # Jagged: apply cut and require at least one passing object
                cut_result = ops[op](arr, value)
                event_mask = ak.to_numpy(ak.any(cut_result, axis=1))
            else:
                event_mask = ak.to_numpy(ops[op](arr, value))
        except Exception:
            event_mask = np.array(ops[op](ak.to_numpy(arr), value))

        mask = mask & event_mask
        n_pass = int(np.sum(mask))
        eff = 100.0 * n_pass / n_total if n_total > 0 else 0.0
        rel_eff = (
            100.0 * n_pass / int(np.sum(mask | ~event_mask))
            if np.sum(mask | ~event_mask) > 0
            else 0.0
        )

        cut_label = f"{branch} {op} {value}"
        cut_flow.append(f"{cut_label:<40} {n_pass:>10} {rel_eff:>10.2f} {eff:>10.2f}")

    n_pass = int(np.sum(mask))
    total_eff = 100.0 * n_pass / n_total if n_total > 0 else 0.0

    result = [
        f"Event selection on: {filepath} ({tree_name})",
        f"Events before cuts: {n_total}",
        f"Events after cuts:  {n_pass}",
        f"Total efficiency:   {total_eff:.2f}%\n",
        "Cut flow:",
    ]
    result.extend(cut_flow)

    return "\n".join(result)


@tool
def compute_invariant_mass(
    pt: list[list[float]],
    eta: list[list[float]],
    phi: list[list[float]],
    mass: list[list[float]],
) -> str:
    """Compute invariant mass from particle four-momenta.

    Takes parallel lists (one per particle) of kinematic arrays.
    For a dimuon invariant mass: provide pt=[mu1_pt, mu2_pt], etc.

    Args:
        pt: Transverse momentum arrays for each particle (list of lists).
        eta: Pseudorapidity arrays for each particle.
        phi: Azimuthal angle arrays for each particle.
        mass: Particle mass arrays for each particle.

    Returns:
        Invariant mass array summary statistics.
    """
    import vector

    if len(pt) < 2:
        return "Need at least 2 particles to compute invariant mass."

    if not all(len(pt[0]) == len(arr) for arr in pt + eta + phi + mass):
        return "All input arrays must have the same length."

    # Build four-vectors and sum
    total = None
    for i in range(len(pt)):
        vec = vector.array(
            {
                "pt": np.array(pt[i]),
                "eta": np.array(eta[i]),
                "phi": np.array(phi[i]),
                "mass": np.array(mass[i]),
            }
        )
        total = vec if total is None else total + vec

    inv_mass = total.mass

    # Filter valid entries
    valid = inv_mass[np.isfinite(inv_mass) & (inv_mass > 0)]

    if len(valid) == 0:
        return "No valid invariant mass values computed."

    lines = [
        "Invariant mass computation:",
        f"  Entries: {len(valid)} (of {len(inv_mass)} total)",
        f"  Mean:   {np.mean(valid):.4f} GeV",
        f"  Median: {np.median(valid):.4f} GeV",
        f"  Std:    {np.std(valid):.4f} GeV",
        f"  Min:    {np.min(valid):.4f} GeV",
        f"  Max:    {np.max(valid):.4f} GeV",
    ]

    # Quick histogram for distribution shape
    counts, edges = np.histogram(valid, bins=20)
    peak_bin = np.argmax(counts)
    peak_center = (edges[peak_bin] + edges[peak_bin + 1]) / 2
    lines.append(f"  Peak at: ~{peak_center:.2f} GeV")

    return "\n".join(lines)


@tool
def compute_delta_r(
    eta1: list[float],
    phi1: list[float],
    eta2: list[float],
    phi2: list[float],
) -> str:
    """Compute ΔR = √(Δη² + Δφ²) between particle pairs.

    Args:
        eta1: η values for first particle collection.
        phi1: φ values for first particle collection.
        eta2: η values for second particle collection.
        phi2: φ values for second particle collection.

    Returns:
        ΔR values summary and distribution.
    """
    eta1_arr = np.array(eta1)
    phi1_arr = np.array(phi1)
    eta2_arr = np.array(eta2)
    phi2_arr = np.array(phi2)

    if len(eta1_arr) != len(eta2_arr):
        return "All input arrays must have the same length."

    deta = eta1_arr - eta2_arr
    dphi = phi1_arr - phi2_arr

    # Wrap dphi to [-π, π]
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))

    delta_r = np.sqrt(deta**2 + dphi**2)

    valid = delta_r[np.isfinite(delta_r)]
    if len(valid) == 0:
        return "No valid ΔR values computed."

    lines = [
        "ΔR computation:",
        f"  Entries: {len(valid)}",
        f"  Mean:   {np.mean(valid):.4f}",
        f"  Median: {np.median(valid):.4f}",
        f"  Std:    {np.std(valid):.4f}",
        f"  Min:    {np.min(valid):.4f}",
        f"  Max:    {np.max(valid):.4f}",
        "",
        "  Distribution:",
    ]

    # Simple text histogram
    counts, edges = np.histogram(valid, bins=10, range=(0, 5))
    max_count = max(counts) if max(counts) > 0 else 1
    for j in range(len(counts)):
        bar_len = int(30 * counts[j] / max_count)
        bar = "█" * bar_len
        lines.append(f"    [{edges[j]:.1f}, {edges[j + 1]:.1f}): {bar} {counts[j]}")

    return "\n".join(lines)


@tool
def fill_histogram(
    data: list[float],
    bins: int = 50,
    hist_range: list[float] | None = None,
    weights: list[float] | None = None,
    label: str | None = None,
) -> str:
    """Create and fill a histogram from array data using boost-histogram.

    Args:
        data: Data values to histogram.
        bins: Number of bins.
        hist_range: [min, max] range. Auto-determined if not provided.
        weights: Event weights (same length as data).
        label: Histogram label/name.

    Returns:
        Histogram bin edges, contents, and statistics (mean, std, integral).
    """
    import hist

    data_arr = np.array(data, dtype=float)
    valid = data_arr[np.isfinite(data_arr)]

    if len(valid) == 0:
        return "No valid data to histogram."

    if hist_range is not None:
        lo, hi = hist_range[0], hist_range[1]
    else:
        lo, hi = float(np.min(valid)), float(np.max(valid))
        # Add small padding
        if lo == hi:
            lo -= 0.5
            hi += 0.5

    h = hist.Hist(hist.axis.Regular(bins, lo, hi, name="x", label=label or "x"))

    w = np.array(weights, dtype=float) if weights is not None else None
    h.fill(valid, weight=w)

    contents = h.values()
    edges = h.axes[0].edges
    integral = float(np.sum(contents))
    bin_width = (hi - lo) / bins

    # Weighted statistics
    if integral > 0:
        centers = (edges[:-1] + edges[1:]) / 2
        mean = float(np.sum(centers * contents) / integral)
        var = float(np.sum((centers - mean) ** 2 * contents) / integral)
        std = float(np.sqrt(var))
    else:
        mean = std = 0.0

    lines = [
        f"Histogram: {label or 'unnamed'}",
        f"  Bins: {bins} | Range: [{lo:.4g}, {hi:.4g}] | Bin width: {bin_width:.4g}",
        f"  Entries: {len(valid)} | Integral: {integral:.6g}",
        f"  Mean: {mean:.4g} | Std: {std:.4g}",
        f"  Underflow: {float(h[hist.underflow]):.4g} | Overflow: {float(h[hist.overflow]):.4g}",
        "",
        "  Bin contents (first 20):",
    ]

    for j in range(min(len(contents), 20)):
        lines.append(f"    [{edges[j]:.4g}, {edges[j + 1]:.4g}): {contents[j]:.4g}")
    if len(contents) > 20:
        lines.append(f"    ... ({len(contents) - 20} more bins)")

    return "\n".join(lines)
