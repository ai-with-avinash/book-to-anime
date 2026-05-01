"""FastAPI application factory.

The CLI launches uvicorn with the app produced here. Tests build their own
app with mock provider factories.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import __version__
from ..parsing import PDFParser
from ..pipeline.video_assembler import FFmpegRunner
from ..state import JobRepository, open_database
from .deps import JobRunner, ProviderFactory
from .routes_jobs import build_job_router
from .routes_sse import build_sse_router

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@dataclass
class AppSettings:
    """Everything ``create_app`` needs to wire the app.

    Attributes:
        data_dir: Root job/state directory (typically platformdirs default,
            overridable by CLI flag / env var).
        provider_factory: Callable producers for the three provider types.
        parser_factory: Defaults to :class:`PDFParser` with stock config; tests
            inject a parser configured for tiny PDFs.
        config_overrides: Anything else a custom CLI run wants exposed via
            ``GET /healthz``.
    """

    data_dir: Path
    provider_factory: ProviderFactory
    parser_factory: Any = field(default=PDFParser)
    config_overrides: dict[str, Any] = field(default_factory=dict)
    ffmpeg_runner: FFmpegRunner | None = None
    ffmpeg_binary: str | None = None


def create_app(settings: AppSettings) -> FastAPI:
    """Build a FastAPI app wired to the given settings.

    The returned app is also usable from ``uvicorn`` directly:
    ``uvicorn booktoanime.api.app:create_app --factory --port 8765``.
    """

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db_path = settings.data_dir / "state.db"
    connection = open_database(db_path)
    repo = JobRepository(connection)
    runner = JobRunner(
        data_dir=settings.data_dir,
        repo=repo,
        provider_factory=settings.provider_factory,
        parser_factory=settings.parser_factory,
        ffmpeg_runner=settings.ffmpeg_runner,
        ffmpeg_binary=settings.ffmpeg_binary,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        try:
            yield
        finally:
            await runner.shutdown()
            connection.close()

    app = FastAPI(
        title="BookToAnime",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    templates_dir = _WEB_DIR / "templates"
    static_dir = _WEB_DIR / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    templates = Jinja2Templates(directory=templates_dir)

    # Inject shared dependencies as app state so route modules can pull them.
    app.state.settings = settings
    app.state.repo = repo
    app.state.runner = runner
    app.state.templates = templates
    app.state.config_overrides = settings.config_overrides

    app.include_router(build_job_router(), prefix="")
    app.include_router(build_sse_router(), prefix="")

    return app
