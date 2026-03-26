"""Data I/O tools — ROOT file reading, NanoAOD, DAS queries, format conversion."""

from __future__ import annotations

import awkward as ak
import numpy as np
import uproot
from langchain_core.tools import tool


@tool
def list_root_contents(filepath: str) -> str:
    """Open a ROOT file and list all objects (TTrees, TH1s, TDirectories) with their types.

    Args:
        filepath: Path to the .root file or xrootd URL.

    Returns:
        Formatted list of object names and class types.
    """
    try:
        f = uproot.open(filepath)
    except Exception as e:
        return f"Error opening ROOT file '{filepath}': {e}"

    lines = []
    for key in f.keys(cycle=False):
        obj = f[key]
        classname = type(obj).__name__
        extra = ""
        if hasattr(obj, "num_entries"):
            extra = f" ({obj.num_entries} entries)"
        elif hasattr(obj, "values"):
            try:
                vals = obj.values()
                extra = f" ({len(vals)} bins)"
            except Exception:
                pass
        lines.append(f"  {key} [{classname}]{extra}")

    if not lines:
        return f"ROOT file '{filepath}' is empty or contains no readable objects."

    header = f"ROOT file: {filepath}\nObjects ({len(lines)}):\n"
    return header + "\n".join(lines)


@tool
def read_root_branches(
    filepath: str,
    tree_name: str,
    branches: list[str],
    max_entries: int = 10000,
    library: str = "ak",
) -> str:
    """Read specified branches from a TTree and return summary statistics.

    Args:
        filepath: Path to the ROOT file.
        tree_name: Name of the TTree.
        branches: Branch names to read.
        max_entries: Maximum number of entries to read.
        library: Output library — "np" for NumPy, "ak" for Awkward.

    Returns:
        Summary statistics (count, mean, std, min, max) for each branch.
    """
    try:
        f = uproot.open(filepath)
        tree = f[tree_name]
    except Exception as e:
        return f"Error reading tree '{tree_name}' from '{filepath}': {e}"

    available = tree.keys()
    missing = [b for b in branches if b not in available]
    if missing:
        return f"Branches not found: {missing}\nAvailable branches: {available[:50]}" + (
            " ..." if len(available) > 50 else ""
        )

    try:
        arrays = tree.arrays(branches, library=library, entry_stop=max_entries)
    except Exception as e:
        return f"Error reading branches: {e}"

    lines = [
        f"Tree: {tree_name} | Entries read: {min(max_entries, tree.num_entries)} "
        f"(of {tree.num_entries} total)\n"
    ]

    for branch in branches:
        arr = arrays[branch]

        # Flatten jagged arrays for stats
        if library == "ak":
            try:
                flat = ak.flatten(arr, axis=None)
                flat_np = ak.to_numpy(flat)
            except Exception:
                flat_np = ak.to_numpy(arr)
        else:
            flat_np = np.asarray(arr).flatten()

        if len(flat_np) == 0:
            lines.append(f"  {branch}: (empty)")
            continue

        flat_np = flat_np.astype(float)
        lines.append(
            f"  {branch}:\n"
            f"    count: {len(flat_np)}\n"
            f"    mean:  {np.mean(flat_np):.6g}\n"
            f"    std:   {np.std(flat_np):.6g}\n"
            f"    min:   {np.min(flat_np):.6g}\n"
            f"    max:   {np.max(flat_np):.6g}"
        )

    return "\n".join(lines)


@tool
def read_nanoaod(
    filepath: str,
    collections: list[str],
    cuts: str | None = None,
    max_entries: int = 100000,
) -> str:
    """Read CMS NanoAOD files with awareness of NanoAOD branch naming conventions.

    Collections like "Muon", "Electron", "Jet" are expanded to read all
    branches with that prefix (e.g., Muon_pt, Muon_eta, Muon_phi, ...).

    Args:
        filepath: Path to NanoAOD ROOT file.
        collections: NanoAOD collections to read (e.g., ["Muon", "Electron", "Jet"]).
        cuts: Selection string (currently informational — describe desired cuts).
        max_entries: Maximum events to process.

    Returns:
        Summary of loaded collections with entry counts and branch statistics.
    """
    try:
        f = uproot.open(filepath)
    except Exception as e:
        return f"Error opening NanoAOD file '{filepath}': {e}"

    # Find the Events tree
    tree_name = None
    for key in f.keys(cycle=False):
        if key.lower() in ("events", "events;1"):
            tree_name = key
            break
    if tree_name is None:
        tree_name = "Events"

    try:
        tree = f[tree_name]
    except Exception as e:
        return f"Error: TTree 'Events' not found in '{filepath}': {e}"

    all_branches = tree.keys()
    lines = [
        f"NanoAOD file: {filepath}",
        f"Events tree: {tree.num_entries} total entries (reading up to {max_entries})\n",
    ]

    if cuts:
        lines.append(f"Requested cuts: {cuts}\n")

    for collection in collections:
        prefix = f"{collection}_"
        col_branches = [b for b in all_branches if b.startswith(prefix)]

        # Also check for nCollection (e.g., nMuon)
        count_branch = f"n{collection}"
        has_count = count_branch in all_branches

        if not col_branches and not has_count:
            lines.append(f"  {collection}: NOT FOUND in file")
            continue

        lines.append(f"  {collection}: {len(col_branches)} branches")

        if has_count:
            try:
                counts = tree[count_branch].array(entry_stop=max_entries, library="np")
                lines.append(
                    f"    n{collection}: mean={np.mean(counts):.2f}, "
                    f"max={int(np.max(counts))}, "
                    f"events_with_>0: {int(np.sum(counts > 0))}"
                )
            except Exception:
                pass

        # Show key kinematic branches
        for suffix in ["pt", "eta", "phi", "mass", "charge"]:
            bname = f"{prefix}{suffix}"
            if bname in col_branches:
                try:
                    arr = tree[bname].array(entry_stop=min(max_entries, 10000), library="ak")
                    flat = ak.to_numpy(ak.flatten(arr, axis=None)).astype(float)
                    if len(flat) > 0:
                        lines.append(
                            f"    {bname}: "
                            f"mean={np.mean(flat):.4g}, "
                            f"std={np.std(flat):.4g}, "
                            f"range=[{np.min(flat):.4g}, {np.max(flat):.4g}]"
                        )
                except Exception:
                    pass

        # List remaining branches
        other = [
            b
            for b in col_branches
            if not any(b.endswith(s) for s in ["pt", "eta", "phi", "mass", "charge"])
        ]
        if other:
            lines.append(f"    Other branches: {', '.join(other[:15])}")
            if len(other) > 15:
                lines.append(f"    ... and {len(other) - 15} more")

        lines.append("")

    return "\n".join(lines)


@tool
def query_das(query: str, limit: int = 50) -> str:
    """Query the CMS Data Aggregation System (DAS) for datasets, files, runs, and luminosity.

    Args:
        query: DAS query string (e.g., "dataset=/ZeroBias/Run2024A-*/NANOAOD").
        limit: Maximum number of results.

    Returns:
        Formatted DAS query results.
    """
    import httpx

    # Try the public JSON API
    try:
        resp = httpx.get(
            "https://cmsweb.cern.ch/das/makepy",
            params={"query": query, "limit": limit, "format": "json"},
            timeout=30,
            follow_redirects=True,
        )
        if resp.status_code == 200:
            try:
                data = resp.json()
                results = data.get("data", [])
                lines = [f"DAS query: {query}", f"Results: {len(results)}\n"]
                for i, entry in enumerate(results[:limit], 1):
                    if isinstance(entry, dict):
                        name = entry.get("dataset", [{}])
                        if isinstance(name, list) and name:
                            name = name[0].get("name", str(entry))
                        lines.append(f"  {i}. {name}")
                    else:
                        lines.append(f"  {i}. {entry}")
                return "\n".join(lines)
            except Exception:
                pass
    except httpx.HTTPError:
        pass

    # Fallback: construct a dasgoclient-compatible command suggestion
    return (
        f"DAS query: {query}\n\n"
        f"Note: Direct DAS API access requires CMS grid credentials (X.509 proxy).\n"
        f"To run this query, use dasgoclient inside a CMS environment:\n\n"
        f'  dasgoclient --query="{query}" --limit={limit}\n\n'
        f"Or in the Docker + CVMFS environment, this will be executed automatically."
    )


@tool
def convert_format(
    input_path: str,
    output_path: str,
    input_format: str,
    output_format: str,
    tree_name: str | None = None,
) -> str:
    """Convert between HEP file formats (ROOT, Parquet, CSV, HDF5).

    Args:
        input_path: Path to input file.
        output_path: Path for output file.
        input_format: Source format — "root", "parquet", "csv", "hdf5".
        output_format: Target format — "root", "parquet", "csv", "hdf5".
        tree_name: TTree name for ROOT files.

    Returns:
        Confirmation with output file path and entry count.
    """
    import awkward as ak

    try:
        # Read input
        if input_format == "root":
            f = uproot.open(input_path)
            if tree_name is None:
                trees = [k for k in f.keys(cycle=False) if hasattr(f[k], "num_entries")]
                if not trees:
                    return f"No TTrees found in '{input_path}'"
                tree_name = trees[0]
            tree = f[tree_name]
            arrays = tree.arrays(library="ak")
            n_entries = tree.num_entries
        elif input_format == "parquet":
            arrays = ak.from_parquet(input_path)
            n_entries = len(arrays)
        elif input_format == "csv":
            import csv as csv_mod
            from pathlib import Path

            text = Path(input_path).read_text()
            reader = csv_mod.DictReader(text.splitlines())
            rows = list(reader)
            n_entries = len(rows)
            if not rows:
                return "CSV file is empty."
            arrays = ak.Array({k: [float(r[k]) if r[k] else 0.0 for r in rows] for k in rows[0]})
        else:
            return f"Unsupported input format: {input_format}"

        # Write output
        if output_format == "parquet":
            ak.to_parquet(arrays, output_path)
        elif output_format == "root":
            with uproot.recreate(output_path) as out:
                out_tree = tree_name or "Events"
                # Convert to dict of numpy arrays for uproot writing
                data = {}
                for field in ak.fields(arrays):
                    try:
                        data[field] = ak.to_numpy(arrays[field])
                    except Exception:
                        data[field] = ak.to_numpy(ak.flatten(arrays[field]))
                out[out_tree] = data
        elif output_format == "csv":
            import csv as csv_mod

            fields = ak.fields(arrays)
            with open(output_path, "w", newline="") as f:
                writer = csv_mod.writer(f)
                writer.writerow(fields)
                for i in range(min(n_entries, len(arrays))):
                    row = []
                    for field in fields:
                        val = arrays[field][i]
                        if isinstance(val, ak.Array):
                            row.append(str(ak.to_list(val)))
                        else:
                            row.append(str(val))
                    writer.writerow(row)
        else:
            return f"Unsupported output format: {output_format}"

        return (
            f"Converted: {input_path} ({input_format}) → {output_path} ({output_format})\n"
            f"Entries: {n_entries}"
        )

    except Exception as e:
        return f"Conversion error: {e}"


@tool
def read_pdf(filepath: str, pages: str | None = None) -> str:
    """Read and extract text content from a PDF file.

    Useful for reading analysis notes (AN), physics analysis summaries (PAS),
    papers, and documentation.

    Args:
        filepath: Path to the PDF file.
        pages: Page range to read (e.g., "1-5", "3", "10-20"). Reads all if not specified.

    Returns:
        Extracted text content from the PDF.
    """
    from pathlib import Path

    path = Path(filepath)
    if not path.exists():
        return f"File not found: {filepath}"
    if not path.suffix.lower() == ".pdf":
        return f"Not a PDF file: {filepath}"

    try:
        import fitz  # PyMuPDF
    except ImportError:
        return _read_pdf_fallback(filepath, pages)

    try:
        doc = fitz.open(filepath)
        total_pages = len(doc)

        # Parse page range
        if pages:
            start, end = _parse_page_range(pages, total_pages)
        else:
            start, end = 0, total_pages

        lines = [
            f"PDF: {filepath}",
            f"Pages: {end - start} of {total_pages} total\n",
        ]

        for i in range(start, end):
            page = doc[i]
            text = page.get_text().strip()
            if text:
                lines.append(f"--- Page {i + 1} ---")
                lines.append(text)
                lines.append("")

        doc.close()

        if len(lines) <= 3:
            return f"PDF '{filepath}' contains no extractable text (may be scanned/image-based)."

        return "\n".join(lines)

    except Exception as e:
        return f"Error reading PDF '{filepath}': {e}"


def _read_pdf_fallback(filepath: str, pages: str | None) -> str:
    """Fallback PDF reader using pdfplumber or raw text extraction."""
    try:
        import pdfplumber

        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            if pages:
                start, end = _parse_page_range(pages, total_pages)
            else:
                start, end = 0, total_pages

            lines = [f"PDF: {filepath}", f"Pages: {end - start} of {total_pages}\n"]
            for i in range(start, end):
                text = pdf.pages[i].extract_text()
                if text:
                    lines.append(f"--- Page {i + 1} ---")
                    lines.append(text.strip())
                    lines.append("")

            return "\n".join(lines) if len(lines) > 3 else f"No extractable text in '{filepath}'."
    except ImportError:
        return (
            "Cannot read PDF: install PyMuPDF or pdfplumber:\n"
            "  pip install pymupdf\n"
            "  pip install pdfplumber"
        )
    except Exception as e:
        return f"Error reading PDF '{filepath}': {e}"


def _parse_page_range(pages: str, total: int) -> tuple[int, int]:
    """Parse a page range string like '1-5' into (start, end) 0-indexed."""
    pages = pages.strip()
    if "-" in pages:
        parts = pages.split("-", 1)
        start = max(0, int(parts[0]) - 1)
        end = min(total, int(parts[1]))
    else:
        start = max(0, int(pages) - 1)
        end = min(total, start + 1)
    return start, end


@tool
def describe_image(filepath: str) -> str:
    """Read an image file and return metadata and basic analysis.

    Useful for inspecting plots, Feynman diagrams, event displays, and other
    figures from analyses. Returns image dimensions, format, color info, and
    can detect text in the image via OCR if available.

    Args:
        filepath: Path to the image file (PNG, JPG, PDF, SVG, etc.).

    Returns:
        Image metadata and description of contents.
    """
    from pathlib import Path

    path = Path(filepath)
    if not path.exists():
        return f"File not found: {filepath}"

    try:
        from PIL import Image

        img = Image.open(filepath)

        lines = [
            f"Image: {filepath}",
            f"  Format: {img.format or path.suffix.upper().strip('.')}",
            f"  Size: {img.width} x {img.height} pixels",
            f"  Mode: {img.mode}",
        ]

        if img.info:
            dpi = img.info.get("dpi")
            if dpi:
                lines.append(f"  DPI: {dpi}")

        # Color analysis
        if img.mode in ("RGB", "RGBA"):
            import numpy as np

            arr = np.array(img)
            lines.append(
                f"  Mean RGB: ({arr[:, :, 0].mean():.0f}, {arr[:, :, 1].mean():.0f}, {arr[:, :, 2].mean():.0f})"
            )

            # Detect if it looks like a CMS plot (white background, specific aspect ratios)
            white_frac = np.mean(np.all(arr[:, :, :3] > 240, axis=2))
            if white_frac > 0.5:
                lines.append("  Background: predominantly white (likely a plot/figure)")

        # File size
        size_kb = path.stat().st_size / 1024
        if size_kb > 1024:
            lines.append(f"  File size: {size_kb / 1024:.1f} MB")
        else:
            lines.append(f"  File size: {size_kb:.1f} KB")

        img.close()
        return "\n".join(lines)

    except ImportError:
        return "Pillow is not installed. Install with: pip install Pillow"
    except Exception as e:
        return f"Error reading image '{filepath}': {e}"
