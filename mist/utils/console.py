"""Shared Rich console singleton and output helpers for MIST.

All user-facing console output should go through these helpers so that
formatting (colours, section headers, completion messages, warnings, and
errors) is consistent across every pipeline.

Rule for warnings.warn() vs these helpers:
    - Use warnings.warn() for programmatic/library warnings that calling code
      may want to catch (e.g. device fallback, encoder compatibility).
    - Use these helpers for every user-facing message printed during a
      pipeline run.
"""
from rich.console import Console

console = Console()


def print_section_header(title: str) -> None:
    """Print a bold section header with surrounding newlines."""
    console.print(f"\n[bold]{title}[/bold]\n")


def print_info(msg: str) -> None:
    """Print a plain informational message."""
    console.print(msg)


def print_warning(msg: str) -> None:
    """Print a yellow warning message."""
    console.print(f"[yellow]Warning:[/yellow] {msg}")


def print_error(msg: str) -> None:
    """Print a bold-red error message."""
    console.print(f"[bold red]Error:[/bold red] {msg}")


def print_success(msg: str) -> None:
    """Print a green success message."""
    console.print(f"[green]\u2713[/green] {msg}")
