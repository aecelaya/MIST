"""Tests for mist.cli.average_weights_entrypoint."""
import pytest

from mist.cli import average_weights_entrypoint as entry


def test_parse_args_ok():
    """_parse_args returns expected namespace for valid arguments."""
    ns = entry._parse_args(["--weights", "a.pt", "b.pt", "--output", "avg.pt"])
    assert ns.weights == ["a.pt", "b.pt"]
    assert ns.output == "avg.pt"


def test_parse_args_missing_required_raises():
    """_parse_args raises SystemExit when required arguments are absent."""
    with pytest.raises(SystemExit):
        entry._parse_args(["--output", "avg.pt"])  # missing --weights


def test_average_weights_entry_calls_average_and_prints(
    monkeypatch, capsys
):
    """average_weights_entry calls average_fold_weights and prints a summary."""
    called = {}
    monkeypatch.setattr(
        entry,
        "average_fold_weights",
        lambda weights, output_path: called.update(
            {"weights": weights, "output_path": output_path}
        ),
        raising=True,
    )

    entry.average_weights_entry(
        ["--weights", "fold0.pt", "fold1.pt", "fold2.pt", "--output", "out.pt"]
    )

    assert called["weights"] == ["fold0.pt", "fold1.pt", "fold2.pt"]
    assert called["output_path"] == "out.pt"
    captured = capsys.readouterr()
    assert "3" in captured.out
    assert "out.pt" in captured.out
