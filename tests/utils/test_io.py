"""Tests for mist.utils.io."""
from pathlib import Path
from typing import Any
import json
import pytest

# MIST imports.
from mist.utils import io


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    """Writing a dict and reading it back preserves structure and types."""
    path = tmp_path / "config.json"
    payload: dict[str, Any] = {
        "int": 7,
        "float": 3.14,
        "bool_true": True,
        "bool_false": False,
        "none": None,
        "str": "hello",
        "unicode": "Café ☕️ – 東京",
        "list": [1, "two", 3.0, {"k": "v"}],
        "dict": {"nested": {"a": 1, "b": [1, 2, 3]}},
    }

    io.write_json_file(str(path), payload)
    assert path.exists(), "JSON file should be created."

    read_back = io.read_json_file(str(path))
    assert read_back == payload, "Round-trip should preserve data exactly."


def test_write_json_file_overwrites_existing(tmp_path: Path) -> None:
    """write_json_file overwrites an existing file with new content."""
    path = tmp_path / "settings.json"
    first = {"version": 1, "name": "first"}
    second = {"version": 2, "name": "second", "extra": [1, 2, 3]}

    io.write_json_file(str(path), first)
    assert json.loads(path.read_text(encoding="utf-8")) == first

    io.write_json_file(str(path), second)
    assert json.loads(path.read_text(encoding="utf-8")) == second
    assert json.loads(path.read_text(encoding="utf-8")) != first


def test_read_json_file_missing_raises(tmp_path: Path) -> None:
    """Reading a non-existent JSON file raises FileNotFoundError."""
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        _ = io.read_json_file(str(missing))


def test_read_json_file_invalid_json_raises(tmp_path: Path) -> None:
    """Invalid JSON content raises json.JSONDecodeError."""
    path = tmp_path / "bad.json"
    path.write_text("{invalid json", encoding="utf-8")  # Broken JSON.
    with pytest.raises(json.JSONDecodeError):
        _ = io.read_json_file(str(path))


def test_write_json_file_uses_utf8_encoding(tmp_path: Path) -> None:
    """write_json_file writes JSON that round-trips UTF-8 content faithfully.

    Note: json.dump defaults to ensure_ascii=True, so non-ASCII characters
    are escaped in the file. We validate by parsing and comparing the value.
    """
    path = tmp_path / "utf8.json"
    payload = {"msg": "naïve façade — München ✓"}

    io.write_json_file(str(path), payload)

    # The file may or may not end with a newline; don't assume it does.
    text = path.read_text(encoding="utf-8")
    assert text.lstrip().startswith("{")
    assert text.rstrip().endswith("}")

    # Parse and compare exact value to ensure characters are preserved.
    parsed = json.loads(text)
    assert parsed == payload

    # Optional: accept either escaped or raw characters.
    # If you later set ensure_ascii=False, these still pass.
    assert ("\\u00ef" in text) or ("ï" in text)
    assert ("\\u00e7" in text) or ("ç" in text)
    assert ("\\u00fc" in text) or ("ü" in text)
