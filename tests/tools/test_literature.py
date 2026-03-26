"""Tests for literature and knowledge tools."""

from __future__ import annotations

import pytest

from src.tools.literature import (
    lookup_particle,
    search_inspire,
)


def test_lookup_particle_known():
    result = lookup_particle.invoke({"name": "Z0"})
    assert "Z" in result
    assert "Mass" in result or "mass" in result.lower()


def test_lookup_particle_alias():
    result = lookup_particle.invoke({"name": "muon"})
    assert "mu" in result.lower()


def test_lookup_particle_unknown():
    result = lookup_particle.invoke({"name": "unobtanium"})
    assert "not found" in result.lower() or "error" in result.lower()


@pytest.mark.integration
def test_search_inspire():
    result = search_inspire.invoke(
        {
            "query": "find t higgs boson CMS",
            "max_results": 3,
        }
    )
    assert "INSPIRE" in result
