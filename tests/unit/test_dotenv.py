"""Unit tests for the tiny .env loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from booktoanime._dotenv import load_dotenv


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with a clean slate for the keys it touches."""

    for key in (
        "BOOKTOANIME_TEST_KEY",
        "BOOKTOANIME_TEST_QUOTED",
        "BOOKTOANIME_TEST_PRESET",
    ):
        monkeypatch.delenv(key, raising=False)


def test_missing_file_is_silent_noop(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "nope.env") == 0


def test_loads_keys(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "BOOKTOANIME_TEST_KEY=hello\n# comment\nBOOKTOANIME_TEST_QUOTED=\"with spaces\"\n",
        encoding="utf-8",
    )
    assert load_dotenv(path) == 2
    assert os.environ["BOOKTOANIME_TEST_KEY"] == "hello"
    assert os.environ["BOOKTOANIME_TEST_QUOTED"] == "with spaces"


def test_existing_env_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOOKTOANIME_TEST_PRESET", "from_shell")
    path = tmp_path / ".env"
    path.write_text("BOOKTOANIME_TEST_PRESET=from_file\n", encoding="utf-8")
    set_count = load_dotenv(path)
    assert set_count == 0
    assert os.environ["BOOKTOANIME_TEST_PRESET"] == "from_shell"


def test_strips_quotes_and_skips_blank(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "\n"
        "  \n"
        "# leading comment\n"
        "BOOKTOANIME_TEST_KEY='single-quoted'\n"
        "no_equals_line\n",
        encoding="utf-8",
    )
    assert load_dotenv(path) == 1
    assert os.environ["BOOKTOANIME_TEST_KEY"] == "single-quoted"
