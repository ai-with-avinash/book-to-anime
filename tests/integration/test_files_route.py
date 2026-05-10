"""Tests for the /api/jobs/{job_id}/files/{filename} route.

Covers the path-traversal defense (allow-listed top-level files plus a tightly
scoped chapter regex), the ``..`` rejection, the symlink-escape defense, and
the 404-when-absent path. The route is otherwise exercised end-to-end by the
chapter-playback view.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from booktoanime.api import AppSettings, ProviderFactory, create_app
from booktoanime.parsing import PDFParser
from booktoanime.pipeline.manifest import ProvidersConfig
from booktoanime.state import JobRepository, open_database


def _never_called() -> Any:
    raise AssertionError("provider factory should not be invoked in files-route tests")


@pytest.fixture
def app_client(tmp_path: Path) -> Iterator[tuple[TestClient, Path]]:
    data_dir = tmp_path / "data"
    factory = ProviderFactory(
        language_factory=cast(Any, _never_called),
        audio_factory=cast(Any, _never_called),
        visual_factory=cast(Any, _never_called),
    )
    settings = AppSettings(
        data_dir=data_dir,
        provider_factory=factory,
        parser_factory=PDFParser,
        config_overrides={
            "providers_obj": ProvidersConfig(language="stub", audio="stub", visual="stub"),
            "providers": {"language": "stub", "audio": "stub", "visual": "stub"},
        },
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, data_dir


def _seed_job(data_dir: Path, job_id: str) -> Path:
    """Create a minimal jobs row + on-disk artifacts so the route accepts the id."""

    data_dir.mkdir(parents=True, exist_ok=True)
    db = open_database(data_dir / "state.db")
    repo = JobRepository(db)
    job_dir = data_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    source = job_dir / "source.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    repo.create_job(
        job_id=job_id,
        source_pdf=source,
        data_dir=data_dir,
        config={"providers": {}},
    )
    db.close()
    return job_dir


def test_serves_top_level_output_mp4(app_client: tuple[TestClient, Path]) -> None:
    client, data_dir = app_client
    job_id = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    job_dir = _seed_job(data_dir, job_id)
    (job_dir / "output.mp4").write_bytes(b"\x00\x00\x00 ftypisomdata")

    response = client.get(f"/api/jobs/{job_id}/files/output.mp4")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.content.startswith(b"\x00\x00\x00 ftyp")


def test_serves_per_chapter_mp4(app_client: tuple[TestClient, Path]) -> None:
    client, data_dir = app_client
    job_id = "AAAAAAAAAAAAAAAAAAAAAAAAAA"
    job_dir = _seed_job(data_dir, job_id)
    chapters = job_dir / "chapters"
    chapters.mkdir()
    (chapters / "chapter_001.mp4").write_bytes(b"chapter-bytes")

    response = client.get(f"/api/jobs/{job_id}/files/chapter_001.mp4")
    assert response.status_code == 200
    assert response.content == b"chapter-bytes"


def test_404_when_file_not_produced(app_client: tuple[TestClient, Path]) -> None:
    client, data_dir = app_client
    job_id = "BBBBBBBBBBBBBBBBBBBBBBBBBB"
    _seed_job(data_dir, job_id)
    response = client.get(f"/api/jobs/{job_id}/files/output.mp4")
    assert response.status_code == 404


def test_unknown_filename_rejected(app_client: tuple[TestClient, Path]) -> None:
    client, data_dir = app_client
    job_id = "CCCCCCCCCCCCCCCCCCCCCCCCCC"
    _seed_job(data_dir, job_id)
    response = client.get(f"/api/jobs/{job_id}/files/secret.json")
    assert response.status_code == 404


def test_unknown_job_returns_404(app_client: tuple[TestClient, Path]) -> None:
    client, _ = app_client
    response = client.get("/api/jobs/MISSINGJOBIDXXXXXXXXXXXXXX/files/output.mp4")
    assert response.status_code == 404


def test_dotdot_traversal_rejected(app_client: tuple[TestClient, Path]) -> None:
    """Starlette normally collapses ``..`` at the routing layer, so this
    arrives as a literal filename — rejected by the allow-list / regex."""

    client, data_dir = app_client
    job_id = "DDDDDDDDDDDDDDDDDDDDDDDDDD"
    _seed_job(data_dir, job_id)
    response = client.get(
        f"/api/jobs/{job_id}/files/..%2F..%2Fetc%2Fpasswd"
    )
    assert response.status_code in (400, 404)


def test_symlink_escape_blocked(app_client: tuple[TestClient, Path], tmp_path: Path) -> None:
    """Symlink under chapters/ that points outside the job dir must not serve."""

    client, data_dir = app_client
    job_id = "EEEEEEEEEEEEEEEEEEEEEEEEEE"
    job_dir = _seed_job(data_dir, job_id)
    chapters = job_dir / "chapters"
    chapters.mkdir()
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"top-secret")
    link = chapters / "chapter_001.mp4"
    try:
        os.symlink(secret, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")

    response = client.get(f"/api/jobs/{job_id}/files/chapter_001.mp4")
    # Either rejected by the relative_to check (400) or the file simply isn't
    # exposed. We must NOT see the secret bytes.
    assert response.status_code in (400, 404)
    assert b"top-secret" not in response.content
