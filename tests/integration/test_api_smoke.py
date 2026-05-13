"""End-to-end API + SSE smoke test using FastAPI TestClient + mocked providers."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from PIL import Image

from booktoanime.api import AppSettings, ProviderFactory, create_app
from booktoanime.parsing import PDFParser
from booktoanime.pipeline.manifest import ProvidersConfig
from booktoanime.providers.base import (
    AudioProvider,
    GeneratedAudio,
    GeneratedImage,
    ImageExplanation,
    LanguageProvider,
    VisualProvider,
)

# --------------------------------------------------------------- fakes


class _FakeLanguage(LanguageProvider):
    name = "fake"

    async def complete(self, request: Any) -> str:
        return json.dumps(
            {
                "summary": "Topic summary one. Two short sentences here. Three closes it out.",
                "key_points": ["a", "b"],
                "estimated_words": 25,
            }
        )

    async def explain_image(self, image: Any, **_: Any) -> ImageExplanation:
        return ImageExplanation(summary="x", detail="y")

    async def close(self) -> None:
        return None


class _FakeAudio(AudioProvider):
    name = "fake_audio"

    async def list_voices(self, language: str | None = None) -> Sequence[str]:
        return ("fake_voice",)

    async def synthesize(self, request: Any, out_path: Path) -> GeneratedAudio:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = 24_000
        duration = max(0.2, min(2.0, len(request.text.split()) * 0.2))
        sf.write(str(out_path), np.zeros(int(sample_rate * duration), dtype=np.float32), sample_rate, "PCM_16")
        return GeneratedAudio(path=out_path, duration_seconds=duration, sample_rate=sample_rate)

    async def close(self) -> None:
        return None


class _FakeVisual(VisualProvider):
    name = "fake_visual"

    def __init__(self, persona_dir: Path) -> None:
        self._persona_dir = persona_dir

    async def prepare(self, *, panel_style: str, narrator_seed: int) -> Path:
        self._persona_dir.mkdir(parents=True, exist_ok=True)
        path = self._persona_dir / f"{panel_style}__{narrator_seed}.png"
        if not path.is_file():
            Image.new("RGB", (32, 32), (10, 30, 60)).save(path)
        return path

    async def render(self, request: Any, out_path: Path) -> GeneratedImage:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (request.width, request.height), (40, 80, 160)).save(out_path)
        return GeneratedImage(
            path=out_path,
            seed=request.seed,
            width=request.width,
            height=request.height,
        )

    async def close(self) -> None:
        return None


# --------------------------------------------------------------- fixtures


async def _stub_ffmpeg_runner(argv, log_path):
    out_path = Path(argv[-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"\x00\x00\x00 ftypisom" + b"X" * 64)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("stub ffmpeg ok", encoding="utf-8")


@pytest.fixture
def app_client(tmp_path: Path) -> Iterator[tuple[TestClient, Path]]:
    data_dir = tmp_path / "data"
    persona_dir = tmp_path / "personas"

    factory = ProviderFactory(
        language_factory=lambda: _FakeLanguage(),
        audio_factory=lambda: _FakeAudio(),
        visual_factory=lambda: _FakeVisual(persona_dir),
    )
    settings = AppSettings(
        data_dir=data_dir,
        provider_factory=factory,
        parser_factory=PDFParser,
        config_overrides={
            "providers_obj": ProvidersConfig(
                language="fake", audio="fake_audio", visual="fake_visual"
            ),
            "providers": {"language": "fake", "audio": "fake_audio", "visual": "fake_visual"},
        },
        ffmpeg_runner=_stub_ffmpeg_runner,
        ffmpeg_binary="ffmpeg-stub",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, data_dir


# --------------------------------------------------------------- happy path


def test_index_renders_html(app_client: tuple[TestClient, Path]) -> None:
    client, _ = app_client
    response = client.get("/")
    assert response.status_code == 200
    assert "StudyPanels" in response.text
    assert "Upload a PDF" in response.text


def test_healthz_returns_version(app_client: tuple[TestClient, Path]) -> None:
    client, _ = app_client
    response = client.get("/api/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_create_job_uploads_pdf_and_returns_id(
    app_client: tuple[TestClient, Path], tiny_pdf: Path
) -> None:
    client, data_dir = app_client
    with tiny_pdf.open("rb") as handle:
        response = client.post(
            "/api/jobs",
            files={"pdf": ("tiny.pdf", handle, "application/pdf")},
            data={"voice_id": "fake_voice"},
        )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "running"
    job_id = body["job_id"]
    assert (data_dir / "jobs" / job_id / "source.pdf").is_file()
    assert (data_dir / "jobs" / job_id / "manifest.json").is_file()


def test_full_run_completes_via_api(
    app_client: tuple[TestClient, Path], tiny_pdf: Path
) -> None:
    client, data_dir = app_client
    with tiny_pdf.open("rb") as handle:
        create = client.post(
            "/api/jobs",
            files={"pdf": ("tiny.pdf", handle, "application/pdf")},
            data={"voice_id": "fake_voice"},
        )
    job_id = create.json()["job_id"]

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status_body = client.get(f"/api/jobs/{job_id}").json()
        if status_body["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    else:
        pytest.fail("job never completed")

    assert status_body["status"] == "completed", status_body
    for stage in (
        "parsing",
        "structuring",
        "style_seeding",
        "storyboard",
        "images",
        "audio",
        "assembly",
    ):
        assert status_body["stages"][stage] == "completed", stage

    job_dir = data_dir / "jobs" / job_id
    assert (job_dir / "structured.json").is_file()
    assert (job_dir / "storyboard.json").is_file()
    assert (job_dir / "images" / "index.json").is_file()
    assert (job_dir / "audio" / "index.json").is_file()
    # Assembly artifacts.
    assert (job_dir / "output.mp4").is_file()
    assert (job_dir / "output.srt").is_file()
    # Style seeding writes a single per-job anchor used by IP-Adapter on
    # SDXL fallback shots.
    style_dir = job_dir / "style"
    assert style_dir.is_dir()
    assert any(p.suffix == ".png" for p in style_dir.iterdir())


def test_unknown_job_returns_404(app_client: tuple[TestClient, Path]) -> None:
    client, _ = app_client
    response = client.get("/api/jobs/DOES_NOT_EXIST")
    assert response.status_code == 404


def test_resume_404_for_missing_job(app_client: tuple[TestClient, Path]) -> None:
    client, _ = app_client
    response = client.post("/api/jobs/MISSING/resume")
    assert response.status_code == 404


def test_sse_404_when_no_live_job(app_client: tuple[TestClient, Path]) -> None:
    client, _ = app_client
    response = client.get("/api/jobs/NEVER_EXISTED/events")
    assert response.status_code == 404
