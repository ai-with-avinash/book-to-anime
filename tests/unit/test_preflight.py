"""Unit tests for the `booktoanime check` preflight probes.

Covers the four probes used by ``cli._run_preflight``:

1. Ollama reachable + configured model present  -> ok
2. Ollama reachable but configured model missing -> fail (model-missing detail)
3. Ollama unreachable (httpx raises)             -> fail (connection detail)
4. Kokoro weights cache absent                   -> fail
5. ffmpeg binary missing                         -> fail
6. tesseract binary missing                      -> fail
7. All four probes green                         -> every result.ok is True

All HTTP and binary lookups are mocked. No network, no filesystem reach
outside ``tmp_path``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from booktoanime.cli import (
    _probe_binary,
    _probe_kokoro,
    _probe_ollama,
    _ProbeResult,
    _run_preflight,
)

# --------------------------------------------------------------------------- helpers


def _ollama_config(model: str = "llama3.1:8b") -> Mapping[str, Any]:
    """Build a minimal config that asks the probe to check a specific model."""

    return {
        "language": {
            "active": "openai_compatible",
            "openai_compatible": {
                "base_url": "http://localhost:11434/v1",
                "model": model,
            },
        },
    }


def _mock_tags_response(model_names: list[str]) -> MagicMock:
    """Mimic the subset of ``httpx.Response`` the probe uses."""

    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {"models": [{"name": name} for name in model_names]}
    return response


# --------------------------------------------------------------------------- Ollama


def test_ollama_reachable_with_configured_model_present() -> None:
    """Probe is happy when the configured model appears in /api/tags."""

    with patch(
        "booktoanime.cli.httpx.get",
        return_value=_mock_tags_response(["llama3.1:8b", "qwen2.5:7b"]),
    ):
        result = _probe_ollama(_ollama_config("llama3.1:8b"))

    assert isinstance(result, _ProbeResult)
    assert result.name == "ollama"
    assert result.ok is True
    assert "llama3.1:8b" in result.detail


def test_ollama_reachable_with_bare_model_name_match() -> None:
    """Probe tolerates a configured ``llama3.1`` (no tag) vs ``llama3.1:8b`` available."""

    with patch(
        "booktoanime.cli.httpx.get",
        return_value=_mock_tags_response(["llama3.1:8b"]),
    ):
        result = _probe_ollama(_ollama_config("llama3.1"))

    assert result.ok is True


def test_ollama_reachable_but_configured_model_missing() -> None:
    """Probe fails with a model-not-pulled message when /api/tags omits it."""

    with patch(
        "booktoanime.cli.httpx.get",
        return_value=_mock_tags_response(["mistral:7b"]),
    ):
        result = _probe_ollama(_ollama_config("llama3.1:8b"))

    assert result.ok is False
    assert "not pulled" in result.detail
    assert "llama3.1:8b" in result.detail


def test_ollama_unreachable_connect_error() -> None:
    """httpx.ConnectError surfaces as a clear 'unreachable' failure."""

    with patch(
        "booktoanime.cli.httpx.get",
        side_effect=httpx.ConnectError("connection refused"),
    ):
        result = _probe_ollama(_ollama_config())

    assert result.ok is False
    assert "unreachable" in result.detail
    assert "ConnectError" in result.detail


def test_ollama_unreachable_timeout() -> None:
    """Timeouts (subclass of httpx.HTTPError) also surface as unreachable."""

    with patch(
        "booktoanime.cli.httpx.get",
        side_effect=httpx.ConnectTimeout("timed out"),
    ):
        result = _probe_ollama(_ollama_config())

    assert result.ok is False
    assert "unreachable" in result.detail


def test_ollama_non_200_response_fails() -> None:
    """Any non-200 status -> probe fails with the status code in the detail."""

    bad = MagicMock(spec=httpx.Response)
    bad.status_code = 502
    with patch("booktoanime.cli.httpx.get", return_value=bad):
        result = _probe_ollama(_ollama_config())

    assert result.ok is False
    assert "502" in result.detail


def test_ollama_skips_model_check_when_no_ollama_shaped_provider() -> None:
    """Reachable Ollama + no Ollama-style provider in config -> ok, model check skipped."""

    config: Mapping[str, Any] = {
        "language": {
            "active": "anthropic",
            "anthropic": {"model": "claude-sonnet-4-6"},
        },
    }
    with patch(
        "booktoanime.cli.httpx.get",
        return_value=_mock_tags_response([]),
    ):
        result = _probe_ollama(config)

    assert result.ok is True
    assert "skipping model check" in result.detail


# --------------------------------------------------------------------------- Kokoro


def test_kokoro_weights_dir_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No project cache and no huggingface cache -> probe fails."""

    # Redirect Path.home() into the empty tmp_path so the HF cache check misses too.
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    result = _probe_kokoro(data_dir)

    assert result.ok is False
    assert "no Kokoro weights" in result.detail


def test_kokoro_weights_dir_present_in_project_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Project-local <data_dir>/models/kokoro/ with a file -> probe passes."""

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    project_cache = tmp_path / "data" / "models" / "kokoro"
    project_cache.mkdir(parents=True)
    (project_cache / "kokoro-v0_19.pth").write_bytes(b"stub")

    result = _probe_kokoro(tmp_path / "data")

    assert result.ok is True
    assert "kokoro" in result.detail


# --------------------------------------------------------------------------- Binaries


def test_ffmpeg_binary_missing() -> None:
    with patch("booktoanime.cli.shutil.which", return_value=None):
        result = _probe_binary("ffmpeg")

    assert result.ok is False
    assert "ffmpeg" in result.detail
    assert "not on PATH" in result.detail


def test_tesseract_binary_missing() -> None:
    with patch("booktoanime.cli.shutil.which", return_value=None):
        result = _probe_binary("tesseract")

    assert result.ok is False
    assert "tesseract" in result.detail
    assert "not on PATH" in result.detail


def test_binary_found() -> None:
    with patch("booktoanime.cli.shutil.which", return_value="/usr/local/bin/ffmpeg"):
        result = _probe_binary("ffmpeg")

    assert result.ok is True
    assert "/usr/local/bin/ffmpeg" in result.detail


# --------------------------------------------------------------------------- Full sweep


def test_run_preflight_all_green(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When every probe is healthy, every returned ``_ProbeResult.ok`` is True."""

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    project_cache = tmp_path / "data" / "models" / "kokoro"
    project_cache.mkdir(parents=True)
    (project_cache / "kokoro-v0_19.pth").write_bytes(b"stub")

    with (
        patch(
            "booktoanime.cli.httpx.get",
            return_value=_mock_tags_response(["llama3.1:8b"]),
        ),
        patch(
            "booktoanime.cli.shutil.which",
            return_value="/usr/local/bin/binary",
        ),
    ):
        results = _run_preflight(_ollama_config("llama3.1:8b"), tmp_path / "data")

    assert len(results) == 4
    names = [r.name for r in results]
    assert names == ["ollama", "kokoro", "ffmpeg", "tesseract"]
    assert all(r.ok for r in results), [
        (r.name, r.detail) for r in results if not r.ok
    ]
