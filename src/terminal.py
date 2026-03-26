"""Open Analysis Agent — interactive terminal interface.

White-labeled Deep Agents terminal for CMS physics analysis.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from src import __version__

# CMS-inspired color theme
OAA_THEME = Theme(
    {
        "oaa.title": "bold bright_blue",
        "oaa.accent": "bright_cyan",
        "oaa.user": "green",
        "oaa.agent": "bright_blue",
        "oaa.tool": "yellow",
        "oaa.error": "bold red",
        "oaa.dim": "dim white",
        "oaa.success": "bold green",
    }
)

BANNER = r"""
   ___                      _                _           _
  / _ \ _ __   ___ _ __    / \   _ __   __ _| |_   _ ___(_)___
 | | | | '_ \ / _ \ '_ \  / _ \ | '_ \ / _` | | | | / __| / __|
 | |_| | |_) |  __/ | | |/ ___ \| | | | (_| | | |_| \__ \ \__ \
  \___/| .__/ \___|_| |_/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
       |_|           _                    _     |___/
                    / \   __ _  ___ _ __ | |_
                   / _ \ / _` |/ _ \ '_ \| __|
                  / ___ \ (_| |  __/ | | | |_
                 /_/   \_\__, |\___|_| |_|\__|
                         |___/
"""

WELCOME_TEXT = """\
**Open Analysis Agent** v{version} — CMS Experiment @ CERN
An AI-powered research assistant for CMS physics analysis at the LHC.

Environment: **{mode}** mode | Model: **{model}**

Type your analysis request, or try:
  • `Open the ROOT file at /data/nano.root and list its contents`
  • `Search INSPIRE for recent CMS Higgs measurements`
  • `Build a pyhf model from signal and background histograms`

Type **exit** or **quit** to end the session. Press **Ctrl+C** to interrupt.
"""


def create_banner_panel(console: Console, mode: str, model: str) -> Panel:
    """Create the startup banner panel."""
    welcome = WELCOME_TEXT.format(version=__version__, mode=mode, model=model)
    return Panel(
        Markdown(welcome),
        title="[oaa.title]Open Analysis Agent[/]",
        subtitle=f"[oaa.dim]v{__version__}[/]",
        border_style="bright_blue",
        padding=(1, 2),
    )


def print_tool_call(console: Console, name: str, args: dict) -> None:
    """Print a tool invocation."""
    args_short = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
    if len(args) > 3:
        args_short += ", ..."
    console.print(f"  [oaa.tool]⚙ {name}[/]({args_short})")


def print_agent_message(console: Console, content: str) -> None:
    """Print an agent response with markdown rendering."""
    console.print()
    console.print(Markdown(content))
    console.print()


def run_interactive(model: str | None = None) -> None:
    """Run the interactive terminal agent session."""
    from langchain_core.messages import HumanMessage

    from src.config import load_config
    from src.environment import detect_environment

    console = Console(theme=OAA_THEME)

    config = load_config()
    env = detect_environment()
    resolved_model = model or config.model.model_string

    # Print banner
    console.print(Text(BANNER, style="bright_blue"))
    console.print(create_banner_panel(console, env["mode"], resolved_model))
    console.print()

    # Create agent
    try:
        from src.agent import create_cms_agent

        agent = create_cms_agent(model=resolved_model, config=config)
    except Exception as e:
        console.print(f"[oaa.error]Failed to initialize agent: {e}[/]")
        console.print("[oaa.dim]Check your API key and model configuration.[/]")
        sys.exit(1)

    messages: list = []

    while True:
        try:
            console.print("[oaa.user]You ▶[/] ", end="")
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[oaa.dim]Session ended.[/]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "/quit", "/exit"):
            console.print("[oaa.dim]Session ended. Good luck with your analysis![/]")
            break

        messages.append(HumanMessage(content=user_input))

        try:
            with console.status("[oaa.accent]Thinking...", spinner="dots"):
                result = agent.invoke({"messages": messages})

            # Process result messages
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print_tool_call(console, tc["name"], tc["args"])

                if hasattr(msg, "content") and msg.content and msg.type == "ai":
                    if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                        print_agent_message(console, msg.content)

            # Update messages for next turn
            messages = result["messages"]

        except KeyboardInterrupt:
            console.print("\n[oaa.dim]Interrupted.[/]")
        except Exception as e:
            console.print(f"[oaa.error]Error: {e}[/]")


def main() -> None:
    """Entry point for the Open Analysis Agent terminal."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="oaa",
        description="Open Analysis Agent — AI research assistant for CMS physics analysis",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help='Model to use (e.g., "anthropic:claude-sonnet-4-20250514")',
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Open Analysis Agent v{__version__}",
    )

    args = parser.parse_args()
    run_interactive(model=args.model)


if __name__ == "__main__":
    main()
