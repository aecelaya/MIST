"""Tests for mist.cli.rank_entrypoint."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mist.cli import rank_entrypoint as entry


# ---------------------------------------------------------------------------
# _parse_rank_args
# ---------------------------------------------------------------------------

class TestParseRankArgs:
    """Tests for rank_entrypoint._parse_rank_args."""

    def test_required_args_are_parsed(self, tmp_path):
        """All required flags are captured in the returned Namespace."""
        ns = entry._parse_rank_args([
            "--results", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"),
            "--output-csv", str(tmp_path / "out.csv"),
        ])
        assert ns.results == [
            str(tmp_path / "a.csv"), str(tmp_path / "b.csv")
        ]
        assert ns.output_csv == str(tmp_path / "out.csv")

    def test_optional_defaults(self, tmp_path):
        """Optional arguments have the documented defaults."""
        ns = entry._parse_rank_args([
            "--results", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"),
            "--output-csv", str(tmp_path / "out.csv"),
        ])
        assert ns.names is None
        assert ns.output_detailed_csv is None
        assert ns.metric_direction_overrides is None
        assert ns.id_column == "id"

    def test_names_parsed_when_supplied(self, tmp_path):
        """--names captures the supplied labels in order."""
        ns = entry._parse_rank_args([
            "--results", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"),
            "--names", "modelA", "modelB",
            "--output-csv", str(tmp_path / "out.csv"),
        ])
        assert ns.names == ["modelA", "modelB"]

    def test_detailed_output_path_parsed(self, tmp_path):
        """--output-detailed-csv is captured when provided."""
        ns = entry._parse_rank_args([
            "--results", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"),
            "--output-csv", str(tmp_path / "out.csv"),
            "--output-detailed-csv", str(tmp_path / "detailed.csv"),
        ])
        assert ns.output_detailed_csv == str(tmp_path / "detailed.csv")

    def test_overrides_path_parsed(self, tmp_path):
        """--metric-direction-overrides is captured when provided."""
        ns = entry._parse_rank_args([
            "--results", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"),
            "--output-csv", str(tmp_path / "out.csv"),
            "--metric-direction-overrides", str(tmp_path / "ov.json"),
        ])
        assert ns.metric_direction_overrides == str(tmp_path / "ov.json")

    def test_id_column_parsed(self, tmp_path):
        """--id-column overrides the default 'id'."""
        ns = entry._parse_rank_args([
            "--results", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"),
            "--output-csv", str(tmp_path / "out.csv"),
            "--id-column", "patient_id",
        ])
        assert ns.id_column == "patient_id"

    def test_missing_required_arg_exits(self, tmp_path):
        """Omitting --output-csv triggers SystemExit."""
        with pytest.raises(SystemExit):
            entry._parse_rank_args([
                "--results", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"),
                # --output-csv missing
            ])


# ---------------------------------------------------------------------------
# _ensure_output_dir
# ---------------------------------------------------------------------------

class TestEnsureOutputDir:
    """Tests for rank_entrypoint._ensure_output_dir."""

    def test_creates_parent_directory(self, tmp_path):
        """Creates nested parent directories for the output CSV."""
        out = tmp_path / "nested" / "out.csv"
        entry._ensure_output_dir(out)
        assert out.parent.is_dir()

    def test_existing_directory_is_noop(self, tmp_path):
        """Calling on an already-existing parent directory does not raise."""
        out = tmp_path / "out.csv"
        entry._ensure_output_dir(out)
        assert tmp_path.is_dir()


# ---------------------------------------------------------------------------
# run_rank
# ---------------------------------------------------------------------------

def _write_results_csv(path: Path, ids, **cols):
    """Write a small results CSV used as input to mist_rank."""
    pd.DataFrame({"id": ids, **cols}).to_csv(path, index=False)


def _make_ns(
    tmp_path: Path,
    *,
    results,
    output_csv,
    names=None,
    output_detailed_csv=None,
    metric_direction_overrides=None,
    id_column="id",
):
    """Build a Namespace mirroring _parse_rank_args output."""
    return argparse.Namespace(
        results=[str(p) for p in results],
        names=names,
        output_csv=str(output_csv),
        output_detailed_csv=(
            None if output_detailed_csv is None else str(output_detailed_csv)
        ),
        metric_direction_overrides=(
            None if metric_direction_overrides is None
            else str(metric_direction_overrides)
        ),
        id_column=id_column,
    )


class TestRunRank:
    """Tests for rank_entrypoint.run_rank."""

    def test_writes_summary_csv(self, tmp_path):
        """run_rank writes a sorted summary CSV with strategy/average_rank."""
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        _write_results_csv(a, ids=["p1", "p2"], WT_dice=[0.9, 0.8])
        _write_results_csv(b, ids=["p1", "p2"], WT_dice=[0.5, 0.4])

        out = tmp_path / "out" / "ranked.csv"
        entry.run_rank(_make_ns(
            tmp_path, results=[a, b], output_csv=out, names=["a", "b"]
        ))

        df = pd.read_csv(out)
        assert list(df.columns) == ["strategy", "average_rank"]
        assert df.iloc[0]["strategy"] == "a"

    def test_default_names_use_file_stems(self, tmp_path):
        """Without --names the file stems become strategy labels."""
        a = tmp_path / "modelA.csv"
        b = tmp_path / "modelB.csv"
        _write_results_csv(a, ids=["p1"], WT_dice=[0.9])
        _write_results_csv(b, ids=["p1"], WT_dice=[0.5])

        out = tmp_path / "ranked.csv"
        entry.run_rank(_make_ns(tmp_path, results=[a, b], output_csv=out))

        df = pd.read_csv(out)
        assert set(df["strategy"]) == {"modelA", "modelB"}

    def test_writes_detailed_csv_when_path_given(self, tmp_path):
        """A detailed CSV is written when --output-detailed-csv is set."""
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        _write_results_csv(
            a, ids=["p1"], WT_dice=[0.9], WT_haus95=[1.0]
        )
        _write_results_csv(
            b, ids=["p1"], WT_dice=[0.5], WT_haus95=[5.0]
        )

        out = tmp_path / "ranked.csv"
        detailed = tmp_path / "deep" / "detailed.csv"
        entry.run_rank(_make_ns(
            tmp_path, results=[a, b],
            output_csv=out, output_detailed_csv=detailed,
            names=["a", "b"],
        ))

        assert detailed.exists()
        df = pd.read_csv(detailed)
        assert {"WT_dice", "WT_haus95"} <= set(df.columns)

    def test_no_detailed_csv_when_not_requested(self, tmp_path):
        """No detailed CSV is created when --output-detailed-csv is omitted."""
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        _write_results_csv(a, ids=["p1"], WT_dice=[0.9])
        _write_results_csv(b, ids=["p1"], WT_dice=[0.5])

        out = tmp_path / "ranked.csv"
        detailed = tmp_path / "detailed.csv"
        entry.run_rank(_make_ns(
            tmp_path, results=[a, b], output_csv=out, names=["a", "b"]
        ))
        assert not detailed.exists()

    def test_overrides_loaded_from_json(self, tmp_path):
        """--metric-direction-overrides is read from JSON and applied."""
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        _write_results_csv(a, ids=["p1"], WT_custom=[10.0])
        _write_results_csv(b, ids=["p1"], WT_custom=[5.0])

        ov = tmp_path / "ov.json"
        ov.write_text(json.dumps({"WT_custom": "higher"}))

        out = tmp_path / "ranked.csv"
        entry.run_rank(_make_ns(
            tmp_path, results=[a, b], output_csv=out,
            names=["a", "b"], metric_direction_overrides=ov,
        ))

        df = pd.read_csv(out)
        # Higher better → a (10.0) wins.
        assert df.iloc[0]["strategy"] == "a"

    def test_overrides_must_be_object(self, tmp_path):
        """A non-object overrides JSON raises ValueError."""
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        _write_results_csv(a, ids=["p1"], WT_dice=[0.9])
        _write_results_csv(b, ids=["p1"], WT_dice=[0.5])

        ov = tmp_path / "ov.json"
        ov.write_text(json.dumps(["not", "a", "dict"]))

        out = tmp_path / "ranked.csv"
        with pytest.raises(ValueError, match="must contain a JSON object"):
            entry.run_rank(_make_ns(
                tmp_path, results=[a, b], output_csv=out,
                names=["a", "b"], metric_direction_overrides=ov,
            ))

    def test_creates_output_directory(self, tmp_path):
        """The parent directory of --output-csv is created if missing."""
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        _write_results_csv(a, ids=["p1"], WT_dice=[0.9])
        _write_results_csv(b, ids=["p1"], WT_dice=[0.5])

        out = tmp_path / "deep" / "nested" / "out.csv"
        entry.run_rank(_make_ns(
            tmp_path, results=[a, b], output_csv=out, names=["a", "b"]
        ))
        assert out.parent.is_dir()
        assert out.exists()

    def test_too_few_results_raises(self, tmp_path):
        """run_rank rejects fewer than two --results paths."""
        a = tmp_path / "a.csv"
        _write_results_csv(a, ids=["p1"], WT_dice=[0.9])

        out = tmp_path / "ranked.csv"
        with pytest.raises(ValueError, match="at least two --results"):
            entry.run_rank(_make_ns(
                tmp_path, results=[a], output_csv=out, names=["a"]
            ))

    def test_names_length_mismatch_raises(self, tmp_path):
        """run_rank rejects a --names list whose length differs."""
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        _write_results_csv(a, ids=["p1"], WT_dice=[0.9])
        _write_results_csv(b, ids=["p1"], WT_dice=[0.5])

        out = tmp_path / "ranked.csv"
        with pytest.raises(ValueError, match="--names has"):
            entry.run_rank(_make_ns(
                tmp_path, results=[a, b], output_csv=out,
                names=["only_one"],
            ))


# ---------------------------------------------------------------------------
# rank_entry (integration)
# ---------------------------------------------------------------------------

class TestRankEntry:
    """Tests for rank_entrypoint.rank_entry."""

    def test_parses_then_runs(self, monkeypatch):
        """rank_entry calls _parse_rank_args then run_rank."""
        ns = SimpleNamespace(
            results=["/a.csv", "/b.csv"],
            names=None,
            output_csv="/out.csv",
            output_detailed_csv=None,
            metric_direction_overrides=None,
            id_column="id",
        )
        called = {"parsed": False, "ran": False}

        monkeypatch.setattr(
            entry, "_parse_rank_args",
            lambda argv=None: (
                called.__setitem__("parsed", True) or ns
            ),
            raising=True,
        )
        monkeypatch.setattr(
            entry, "run_rank",
            lambda n: called.__setitem__("ran", True),
            raising=True,
        )

        entry.rank_entry([
            "--results", "/a.csv", "/b.csv",
            "--output-csv", "/out.csv",
        ])

        assert called["parsed"] is True
        assert called["ran"] is True
