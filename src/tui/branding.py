from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.tui.renderer import clean_text

# Sampled from assets/logo-nontransparent.png. Embedded text keeps installed
# packages independent of the repository's asset paths and imaging libraries.
LOGO = (
    "       ▄███▄▄█▄▄▄",
    "    ▄▄█    ████████▄",
    "  ▄█▀█ ▄▀▄ █▀▀▀▀▀▀▀▀▀",
    "  █  ▀▄▀█  █▄████████▀▀▄",
    "▄█▀▄▄   █  █▀█████▀▀▄███",
    "█▀ ▀▄▀▀▀▀▄▄█▄▄▄▀▄▄█████▀",
    "▀█▄  █▄▄▄▄█████ ████▀",
    "  █  ▀▀   ▀████ ████",
    "  ▀█▄▄█▄█▀█████ ███▀",
    "     █▄    ████ ▀▀",
    "      ▀▀▀▀▀▀▀▀▀",
)


def print_banner(
    console: Console, *, model: str, cwd: Path, window: int, window_assumed: bool
) -> None:
    details = Text("TonCode\n", style="bold")
    details.append(f"Model   {clean_text(model)}\n", style="cyan")
    details.append(
        f"Context {window:,} tokens" + (" (assumed)" if window_assumed else "") + "\n",
        style="dim",
    )
    details.append(f"{clean_text(str(cwd))}\n\n", style="dim")
    details.append(
        "Enter: submit · Alt+Enter: newline\nCtrl+C: interrupt · /help", style="dim"
    )
    encoding = getattr(console.file, "encoding", None) or "utf-8"
    try:
        "█▀▄".encode(encoding)
        unicode_logo = True
    except UnicodeEncodeError:
        unicode_logo = False
    if console.width < 60 or not unicode_logo:
        console.print(details)
        return
    grid = Table.grid(padding=(0, 3))
    grid.add_column(width=24)
    grid.add_column()
    grid.add_row(Text("\n".join(LOGO), style="bold cyan"), details)
    console.print(grid)
    console.print()
