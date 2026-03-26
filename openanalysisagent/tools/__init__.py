"""CMS tool suite for Open Analysis Agent."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from openanalysisagent.tools.analysis import (
    apply_event_selection,
    compute_delta_r,
    compute_invariant_mass,
    fill_histogram,
)
from openanalysisagent.tools.data_io import (
    convert_format,
    describe_image,
    list_root_contents,
    query_das,
    read_nanoaod,
    read_pdf,
    read_root_branches,
)
from openanalysisagent.tools.literature import (
    get_bibtex,
    lookup_particle,
    query_hepdata,
    search_arxiv,
    search_inspire,
    web_search,
)
from openanalysisagent.tools.statistics import (
    build_cabinetry_workspace,
    build_pyhf_model,
    run_cls_limit,
    run_discovery_test,
    run_fit,
)
from openanalysisagent.tools.visualization import (
    plot_brazil_band,
    plot_distribution,
    plot_pull,
)

# Lightweight tools — always available (pure Python / pip-installable)
LIGHTWEIGHT_TOOLS: list[BaseTool] = [
    list_root_contents,
    read_root_branches,
    read_nanoaod,
    query_das,
    convert_format,
    read_pdf,
    describe_image,
    search_inspire,
    get_bibtex,
    search_arxiv,
    query_hepdata,
    lookup_particle,
    web_search,
    apply_event_selection,
    compute_invariant_mass,
    compute_delta_r,
    fill_histogram,
    build_pyhf_model,
    run_cls_limit,
    run_discovery_test,
    run_fit,
    build_cabinetry_workspace,
    plot_distribution,
    plot_brazil_band,
    plot_pull,
]


def get_all_tools(env: dict[str, Any] | None = None) -> list[BaseTool]:
    """Return all available tools based on the runtime environment."""
    tools = list(LIGHTWEIGHT_TOOLS)

    if env and env.get("mode") == "full":
        from openanalysisagent.tools.cmssw import (
            generate_cmsrun_config,
            query_global_tag,
            run_cmsrun,
            setup_cmssw,
        )

        tools.extend([setup_cmssw, run_cmsrun, generate_cmsrun_config, query_global_tag])

    return tools
