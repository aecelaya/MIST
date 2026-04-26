"""Entrypoint for the analyze command."""
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
import argparse

# MIST imports.
from mist.analyze_data.analyzer import Analyzer
from mist.cli import args as argmod


def prepare_analyze_dirs(cli: argparse.Namespace) -> Path:
    """Create output tree for the *analyze* command."""
    results_dir = Path(cli.results or "./results").expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def analyze_entry(argv: list[str] | None = None) -> None:
    """Entrypoint for the analyze command."""
    # Build the argument parser.
    parser = argmod.ArgParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="MIST Analyze pipeline.",
    )
    argmod.add_analyzer_args(parser)

    # Parse CLI (argv is optional for testability).
    cli = parser.parse_args(argv)

    # Ensure output directory structure exists.
    results_dir = prepare_analyze_dirs(cli)

    # Block accidental overwrite of an existing config unless --overwrite.
    config_path = results_dir / "config.json"
    if config_path.exists() and not getattr(cli, "overwrite", False):
        raise FileExistsError(
            f"Found existing configuration at {config_path}. "
            "Use --overwrite to replace it."
        )

    # Run analysis (Analyzer expects the full Namespace).
    analyzer = Analyzer(cli)
    analyzer.run()


if __name__ == "__main__":
    analyze_entry()  # pragma: no cover
