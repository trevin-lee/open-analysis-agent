"""Tests for data I/O tools."""

from __future__ import annotations

import pytest

from src.tools.data_io import (
    convert_format,
    list_root_contents,
    read_nanoaod,
    read_root_branches,
)


def test_list_root_contents(sample_root_file):
    result = list_root_contents.invoke({"filepath": str(sample_root_file)})
    assert "Events" in result
    assert "ROOT file" in result


def test_list_root_contents_missing_file():
    result = list_root_contents.invoke({"filepath": "/nonexistent/file.root"})
    assert "Error" in result


def test_read_root_branches(sample_root_file):
    result = read_root_branches.invoke(
        {
            "filepath": str(sample_root_file),
            "tree_name": "Events",
            "branches": ["nMuon", "Jet_pt"],
            "max_entries": 100,
        }
    )
    assert "nMuon" in result
    assert "mean" in result
    assert "count" in result


def test_read_root_branches_missing_branch(sample_root_file):
    result = read_root_branches.invoke(
        {
            "filepath": str(sample_root_file),
            "tree_name": "Events",
            "branches": ["NonExistentBranch"],
        }
    )
    assert "not found" in result


def test_read_nanoaod(sample_root_file):
    result = read_nanoaod.invoke(
        {
            "filepath": str(sample_root_file),
            "collections": ["Muon", "Jet"],
        }
    )
    assert "NanoAOD file" in result


def test_convert_format_root_to_parquet(sample_root_file, tmp_dir):
    pytest.importorskip("pyarrow")
    output = tmp_dir / "output.parquet"
    result = convert_format.invoke(
        {
            "input_path": str(sample_root_file),
            "output_path": str(output),
            "input_format": "root",
            "output_format": "parquet",
        }
    )
    assert "Converted" in result
    assert output.exists()


def test_convert_format_root_to_csv(sample_root_file, tmp_dir):
    output = tmp_dir / "output.csv"
    result = convert_format.invoke(
        {
            "input_path": str(sample_root_file),
            "output_path": str(output),
            "input_format": "root",
            "output_format": "csv",
        }
    )
    assert "Converted" in result
