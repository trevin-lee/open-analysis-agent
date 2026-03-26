"""Shared fixtures for OpenAnalysisAgent tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_root_file(tmp_dir):
    """Create a small ROOT file with a TTree for testing."""
    import uproot

    filepath = tmp_dir / "test.root"
    rng = np.random.default_rng(42)

    n_events = 1000
    data = {
        "nMuon": rng.poisson(2, n_events).astype(np.int32),
        "Jet_pt": rng.exponential(50, n_events).astype(np.float32),
        "MET_pt": rng.exponential(30, n_events).astype(np.float32),
    }

    # Create jagged muon arrays based on nMuon counts
    muon_pts = []
    muon_etas = []
    muon_phis = []
    for n in data["nMuon"]:
        muon_pts.append(rng.exponential(40, n).astype(np.float32).tolist())
        muon_etas.append(rng.uniform(-2.4, 2.4, n).astype(np.float32).tolist())
        muon_phis.append(rng.uniform(-np.pi, np.pi, n).astype(np.float32).tolist())

    with uproot.recreate(str(filepath)) as f:
        f["Events"] = {
            "nMuon": data["nMuon"],
            "Jet_pt": data["Jet_pt"],
            "MET_pt": data["MET_pt"],
        }

    return filepath


@pytest.fixture
def sample_workspace():
    """Return a simple pyhf workspace dict for testing."""
    return {
        "channels": [
            {
                "name": "SR",
                "samples": [
                    {
                        "name": "signal",
                        "data": [5.0, 10.0, 5.0],
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None},
                        ],
                    },
                    {
                        "name": "background",
                        "data": [50.0, 100.0, 50.0],
                        "modifiers": [
                            {
                                "name": "staterror_SR",
                                "type": "staterror",
                                "data": [7.07, 10.0, 7.07],
                            }
                        ],
                    },
                ],
            }
        ],
        "observations": [{"name": "SR", "data": [48.0, 105.0, 52.0]}],
        "measurements": [
            {
                "name": "measurement",
                "config": {"poi": "mu", "parameters": []},
            }
        ],
        "version": "1.0.0",
    }
