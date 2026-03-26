"""Literature and knowledge tools — INSPIRE-HEP, arXiv, HEPData, PDG lookups."""

from __future__ import annotations

import httpx
from langchain_core.tools import tool


@tool
def search_inspire(
    query: str,
    sort: str = "mostrecent",
    max_results: int = 10,
) -> str:
    """Search the INSPIRE-HEP database for papers by topic, author, collaboration, or arXiv ID.

    Args:
        query: INSPIRE search query (e.g., "find t higgs boson CMS", "a John Smith").
        sort: Sort order — "mostrecent" or "mostcited".
        max_results: Maximum results to return.

    Returns:
        Formatted list with titles, authors, arXiv IDs, citation counts, and DOIs.
    """
    base_url = "https://inspirehep.net/api/literature"
    params = {
        "q": query,
        "sort": sort,
        "size": max_results,
        "fields": "titles,authors,arxiv_eprints,citation_count,dois,publication_info",
    }

    try:
        resp = httpx.get(base_url, params=params, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"INSPIRE search error: {e}"

    hits = data.get("hits", {}).get("hits", [])
    total = data.get("hits", {}).get("total", 0)

    if not hits:
        return f"No results found for query: {query}"

    lines = [f'INSPIRE search: "{query}" ({total} total, showing {len(hits)})\n']

    for i, hit in enumerate(hits, 1):
        meta = hit.get("metadata", {})

        title = "Untitled"
        titles = meta.get("titles", [])
        if titles:
            title = titles[0].get("title", "Untitled")

        # First few authors
        authors = meta.get("authors", [])
        if len(authors) <= 3:
            author_str = ", ".join(a.get("full_name", "?") for a in authors)
        else:
            author_str = (
                ", ".join(a.get("full_name", "?") for a in authors[:3])
                + f" et al. ({len(authors)} authors)"
            )

        arxiv = ""
        eprints = meta.get("arxiv_eprints", [])
        if eprints:
            arxiv = eprints[0].get("value", "")

        citations = meta.get("citation_count", 0)

        doi = ""
        dois = meta.get("dois", [])
        if dois:
            doi = dois[0].get("value", "")

        recid = hit.get("id", "")

        lines.append(f"  {i}. {title}")
        lines.append(f"     Authors: {author_str}")
        if arxiv:
            lines.append(f"     arXiv: {arxiv}")
        if doi:
            lines.append(f"     DOI: {doi}")
        lines.append(f"     Citations: {citations} | INSPIRE ID: {recid}")
        lines.append("")

    return "\n".join(lines)


@tool
def get_bibtex(identifiers: list[str]) -> str:
    """Retrieve BibTeX entries from INSPIRE-HEP for one or more papers.

    Args:
        identifiers: List of INSPIRE record IDs or arXiv IDs.

    Returns:
        BibTeX entries for all requested papers.
    """
    entries = []

    for ident in identifiers:
        # Determine if it's an arXiv ID or INSPIRE recid
        if ident.replace(".", "").replace("/", "").replace("-", "").isdigit() or "." in ident:
            url = f"https://inspirehep.net/api/arxiv/{ident}"
        else:
            url = f"https://inspirehep.net/api/literature/{ident}"

        try:
            resp = httpx.get(
                url,
                headers={"Accept": "application/x-bibtex"},
                timeout=15,
                follow_redirects=True,
            )
            if resp.status_code == 200:
                entries.append(resp.text.strip())
            else:
                entries.append(f"% Error fetching BibTeX for '{ident}': HTTP {resp.status_code}")
        except Exception as e:
            entries.append(f"% Error fetching BibTeX for '{ident}': {e}")

    return "\n\n".join(entries)


@tool
def search_arxiv(
    query: str,
    category: str | None = None,
    max_results: int = 10,
) -> str:
    """Search arXiv for preprints by query, author, or category.

    Args:
        query: Search query.
        category: arXiv category filter (e.g., "hep-ex", "hep-ph").
        max_results: Maximum results.

    Returns:
        Formatted list with titles, authors, abstracts, and arXiv IDs.
    """
    import xml.etree.ElementTree as ET

    search_query = query
    if category:
        search_query = f"cat:{category} AND all:{query}"

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    try:
        resp = httpx.get(
            "http://export.arxiv.org/api/query",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as e:
        return f"arXiv search error: {e}"

    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", ns)

    if not entries:
        return f"No arXiv results for: {query}"

    lines = [f'arXiv search: "{query}"' + (f" [{category}]" if category else "") + "\n"]

    for i, entry in enumerate(entries, 1):
        title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
        arxiv_id = (entry.findtext("atom:id", "", ns) or "").split("/abs/")[-1]

        authors = entry.findall("atom:author", ns)
        if len(authors) <= 3:
            author_str = ", ".join((a.findtext("atom:name", "?", ns) or "?") for a in authors)
        else:
            author_str = (
                ", ".join((a.findtext("atom:name", "?", ns) or "?") for a in authors[:3])
                + f" et al. ({len(authors)} authors)"
            )

        summary = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")
        if len(summary) > 200:
            summary = summary[:200] + "..."

        published = (entry.findtext("atom:published", "", ns) or "")[:10]

        categories = entry.findall("atom:category", ns)
        cats = [c.get("term", "") for c in categories if c.get("term")]

        lines.append(f"  {i}. {title}")
        lines.append(f"     arXiv: {arxiv_id} | Published: {published}")
        lines.append(f"     Authors: {author_str}")
        lines.append(f"     Categories: {', '.join(cats[:5])}")
        lines.append(f"     Abstract: {summary}")
        lines.append("")

    return "\n".join(lines)


@tool
def query_hepdata(inspire_id: str, table_name: str | None = None) -> str:
    """Retrieve published data tables from HEPData for a given INSPIRE record.

    Args:
        inspire_id: INSPIRE record ID.
        table_name: Specific table to retrieve (optional).

    Returns:
        Data table contents with column names, units, and values.
    """
    base_url = f"https://www.hepdata.net/record/ins{inspire_id}"

    try:
        resp = httpx.get(f"{base_url}?format=json", timeout=30, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"HEPData query error for INSPIRE ID {inspire_id}: {e}"

    tables = data.get("tables", data.get("data_tables", []))
    if not tables:
        return f"No data tables found for INSPIRE ID {inspire_id}"

    if table_name:
        matched = [t for t in tables if t.get("name", "") == table_name]
        if not matched:
            available = [t.get("name", "unnamed") for t in tables]
            return f"Table '{table_name}' not found. Available tables: {available}"
        tables = matched

    lines = [f"HEPData record: INSPIRE {inspire_id}\n"]

    for tbl in tables[:10]:
        name = tbl.get("name", "unnamed")
        description = tbl.get("description", "")
        lines.append(f"Table: {name}")
        if description:
            lines.append(f"  Description: {description[:150]}")

        # Get table data
        table_url = tbl.get("url") or tbl.get("data_file")
        if table_url:
            try:
                tbl_resp = httpx.get(table_url, timeout=15, follow_redirects=True)
                if tbl_resp.status_code == 200:
                    tbl_data = tbl_resp.json()
                    indep = tbl_data.get("independent_variables", [])
                    dep = tbl_data.get("dependent_variables", [])

                    for var in indep:
                        header = var.get("header", {})
                        vals = var.get("values", [])
                        lines.append(
                            f"  Independent: {header.get('name', '?')} "
                            f"[{header.get('units', '')}] ({len(vals)} points)"
                        )

                    for var in dep:
                        header = var.get("header", {})
                        vals = var.get("values", [])
                        lines.append(
                            f"  Dependent: {header.get('name', '?')} "
                            f"[{header.get('units', '')}] ({len(vals)} points)"
                        )
                        # Show first few values
                        for v in vals[:5]:
                            val_str = str(v.get("value", "?"))
                            errs = v.get("errors", [])
                            if errs:
                                err_strs = []
                                for err in errs:
                                    if "asymerror" in err:
                                        ae = err["asymerror"]
                                        err_strs.append(
                                            f"+{ae.get('plus', '?')}/{ae.get('minus', '?')}"
                                        )
                                    elif "symerror" in err:
                                        err_strs.append(f"± {err['symerror']}")
                                val_str += " " + " ".join(err_strs)
                            lines.append(f"    {val_str}")
                        if len(vals) > 5:
                            lines.append(f"    ... ({len(vals) - 5} more values)")
            except Exception:
                pass

        lines.append("")

    return "\n".join(lines)


@tool
def lookup_particle(name: str) -> str:
    """Look up particle properties from the PDG database using the particle package.

    Args:
        name: Particle name (e.g., "W+", "Z0", "H", "top", "muon", "pi+").

    Returns:
        Mass, width, lifetime, quantum numbers, and dominant decay modes.
    """
    from particle import Particle

    # Common aliases
    aliases = {
        "H": "H(125)",
        "higgs": "H(125)",
        "Higgs": "H(125)",
        "top": "t",
        "bottom": "b",
        "charm": "c",
        "strange": "s",
        "up": "u",
        "down": "d",
        "muon": "mu-",
        "electron": "e-",
        "tau": "tau-",
        "Z": "Z0",
        "Z boson": "Z0",
        "W": "W+",
        "W boson": "W+",
    }

    search_name = aliases.get(name, name)

    try:
        # Try exact match first
        try:
            p = Particle.from_string(search_name)
        except Exception:
            # Try searching
            matches = Particle.findall(search_name)
            if not matches:
                matches = Particle.findall(name)
            if not matches:
                return (
                    f"Particle '{name}' not found. Try a PDG name like 'W+', 'Z0', 'mu-', 'H(125)'."
                )
            p = matches[0]

        lines = [f"Particle: {p.name} (PDG ID: {p.pdgid})"]

        if p.mass is not None:
            lines.append(f"  Mass: {p.mass:.6g} ± {p.mass_upper:.6g} / {p.mass_lower:.6g} MeV/c²")

        if p.width is not None:
            lines.append(f"  Width: {p.width:.6g} MeV")

        if p.lifetime is not None:
            lines.append(f"  Lifetime: {p.lifetime:.6g} ns")

        if p.charge is not None:
            lines.append(f"  Charge: {p.charge:g} e")

        lines.append(f"  Spin: {p.J}")

        if p.P is not None:
            lines.append(f"  Parity: {p.P:+d}")

        if p.C is not None:
            lines.append(f"  C-parity: {p.C:+d}")

        try:
            anti = Particle.from_pdgid(-p.pdgid)
            if anti and anti.name != p.name:
                lines.append(f"  Antiparticle: {anti.name}")
        except Exception:
            pass

        # LaTeX representation
        if p.latex_name:
            lines.append(f"  LaTeX: {p.latex_name}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error looking up particle '{name}': {e}"


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for physics-related information, documentation, or general queries.

    Uses Tavily search API if TAVILY_API_KEY is set, otherwise falls back to
    DuckDuckGo. Useful for finding CMS TWiki pages, ROOT documentation,
    analysis notes, and general physics references.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        Formatted search results with titles, URLs, and snippets.
    """
    import os

    tavily_key = os.environ.get("TAVILY_API_KEY")

    if tavily_key:
        return _search_tavily(query, max_results, tavily_key)

    return _search_duckduckgo(query, max_results)


def _search_tavily(query: str, max_results: int, api_key: str) -> str:
    """Search using Tavily API."""
    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Tavily search error: {e}"

    results = data.get("results", [])
    if not results:
        return f"No results found for: {query}"

    lines = [f'Web search: "{query}"\n']
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("content", "")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        lines.append(f"  {i}. {title}")
        lines.append(f"     {url}")
        lines.append(f"     {snippet}")
        lines.append("")

    return "\n".join(lines)


def _search_duckduckgo(query: str, max_results: int) -> str:
    """Search using DuckDuckGo HTML (no API key needed)."""
    try:
        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "OpenAnalysisAgent/1.0"},
            timeout=15,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as e:
        return f"Web search error: {e}"

    # Parse results from HTML
    import re

    lines = [f'Web search: "{query}"\n']
    results_found = 0

    # Extract result blocks
    for match in re.finditer(
        r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a class="result__snippet"[^>]*>(.*?)</a>',
        resp.text,
        re.DOTALL,
    ):
        if results_found >= max_results:
            break
        url = match.group(1)
        title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
        snippet = re.sub(r"<[^>]+>", "", match.group(3)).strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        results_found += 1
        lines.append(f"  {results_found}. {title}")
        lines.append(f"     {url}")
        lines.append(f"     {snippet}")
        lines.append("")

    if results_found == 0:
        return f"No results found for: {query}"

    return "\n".join(lines)
