"""Top-level Typer CLI.

``booktoanime run`` starts the FastAPI server. ``booktoanime resume <job_id>``
re-runs a failed job from the last completed stage. ``booktoanime check``
probes the free-stack dependencies (Ollama, Kokoro weights, ffmpeg, tesseract)
and exits non-zero on any missing piece. ``booktoanime version`` prints the
package version.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import threading
import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
import uvicorn
import yaml
from platformdirs import user_data_dir
from rich.console import Console
from rich.table import Table

from . import __version__
from ._dotenv import load_dotenv
from .api import AppSettings, ProviderFactory, create_app
from .errors import BookToAnimeError
from .pipeline.manifest import JobManifest, ProvidersConfig

_console = Console()

app = typer.Typer(
    add_completion=False,
    help="Convert a PDF into a STEM motion-comic narrated explainer video.",
)


def _default_data_dir() -> Path:
    return Path(user_data_dir("booktoanime", appauthor=False))


def _resolve_data_dir(explicit: Path | None) -> Path:
    import os

    if explicit is not None:
        return explicit.expanduser().resolve()
    env_value = os.environ.get("BOOKTOANIME_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return _default_data_dir()


def _load_yaml_config(config_path: Path | None) -> Mapping[str, Any]:
    if config_path is None:
        # Look for `config.yaml` in CWD; missing is fine — first-run gating happens later.
        candidate = Path.cwd() / "config.yaml"
        if not candidate.is_file():
            return {}
        config_path = candidate
    if not config_path.is_file():
        raise typer.BadParameter(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise typer.BadParameter("config root must be a mapping")
    return loaded


def _build_settings(config: Mapping[str, Any], data_dir: Path) -> AppSettings:
    if not config.get("language") or not config.get("audio") or not config.get("visual"):
        _print_first_run_hint()
        raise typer.Exit(code=1)

    factory = ProviderFactory.from_config(config)
    providers_obj = ProvidersConfig(
        language=str(config["language"].get("active", "")),
        audio=str(config["audio"].get("active", "")),
        visual=str(config["visual"].get("active", "")),
    )
    overrides: dict[str, Any] = {
        "providers_obj": providers_obj,
        "providers": providers_obj.model_dump(),
    }
    return AppSettings(
        data_dir=data_dir,
        provider_factory=factory,
        config_overrides=overrides,
    )


def _print_first_run_hint() -> None:
    _console.print(
        "[bold red]No language / audio / visual provider configured.[/bold red]\n"
        "Create a [bold]config.yaml[/bold] in the working directory or pass --config.\n\n"
        "Recommended free-stack path (zero per-book cost):\n"
        "  * [bold]Ollama Llama 3.1 8B[/bold] + Kokoro TTS + local SDXL\n\n"
        "Optional hosted fallbacks (rough per-book costs):\n"
        "  * [bold]Groq Llama 3.3 70B[/bold]: ~$0.05-$0.30 per book\n"
        "  * [bold]Gemini Flash[/bold]: ~$0.10-$0.50 per book\n"
        "  * [bold]Claude Sonnet[/bold]: ~$2-$8 per book\n\n"
        "See `config.example.yaml` in the repository for the full schema."
    )


# --------------------------------------------------------- preflight (check)


@dataclass(frozen=True)
class _ProbeResult:
    name: str
    ok: bool
    detail: str


def _probe_ollama(config: Mapping[str, Any]) -> _ProbeResult:
    """Reach Ollama's /api/tags and assert the configured model is present.

    Honors ``OLLAMA_HOST`` (defaults to ``http://localhost:11434``). When the
    active language provider isn't ``openai_compatible`` or its base_url isn't
    Ollama, we just probe reachability and skip the model assertion.
    """

    import os

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = f"{host}/api/tags"
    expected_model: str | None = None
    language = config.get("language") if isinstance(config, Mapping) else None
    if isinstance(language, Mapping):
        active = str(language.get("active", ""))
        sub = language.get(active) if active else None
        if isinstance(sub, Mapping):
            base_url = str(sub.get("base_url", ""))
            if "11434" in base_url or "ollama" in base_url.lower():
                model = sub.get("model")
                if isinstance(model, str):
                    expected_model = model

    try:
        response = httpx.get(url, timeout=2.0)
    except httpx.HTTPError as exc:
        return _ProbeResult(
            name="ollama",
            ok=False,
            detail=f"unreachable at {url} ({exc.__class__.__name__})",
        )

    if response.status_code != 200:
        return _ProbeResult(
            name="ollama",
            ok=False,
            detail=f"{url} returned HTTP {response.status_code}",
        )

    if expected_model is None:
        return _ProbeResult(
            name="ollama",
            ok=True,
            detail=(
                "reachable; no Ollama-shaped language provider in config "
                "(skipping model check)"
            ),
        )

    try:
        payload = response.json()
    except ValueError:
        return _ProbeResult(
            name="ollama",
            ok=False,
            detail=f"{url} returned non-JSON body",
        )

    models = payload.get("models") if isinstance(payload, dict) else None
    available = {
        str(item.get("name"))
        for item in (models or [])
        if isinstance(item, Mapping) and item.get("name")
    }
    # Ollama tags include the ":tag" suffix; tolerate both with-tag and bare names.
    expected_bare = expected_model.split(":", 1)[0]
    matched = any(
        name == expected_model or name.split(":", 1)[0] == expected_bare for name in available
    )
    if not matched:
        return _ProbeResult(
            name="ollama",
            ok=False,
            detail=(
                f"model {expected_model!r} not pulled. Run "
                f"`ollama pull {expected_model}`."
            ),
        )
    return _ProbeResult(
        name="ollama",
        ok=True,
        detail=f"reachable; model {expected_model!r} present",
    )


def _probe_kokoro(data_dir: Path) -> _ProbeResult:
    """Best-effort Kokoro weights presence check.

    Kokoro's `kokoro` package caches under huggingface's default cache
    (``~/.cache/huggingface``) or a project-specific location. We treat
    presence of either ``<data_dir>/models/kokoro/`` *or* a huggingface
    cache entry for ``hexgrad/Kokoro-82M`` as proof. Failing both, the
    user is told how to prefetch.
    """

    project_cache = data_dir / "models" / "kokoro"
    if project_cache.is_dir() and any(project_cache.iterdir()):
        return _ProbeResult(
            name="kokoro",
            ok=True,
            detail=f"weights cached at {project_cache}",
        )

    home = Path.home()
    candidates = (
        home / ".cache" / "huggingface" / "hub" / "models--hexgrad--Kokoro-82M",
        home / ".cache" / "huggingface" / "models--hexgrad--Kokoro-82M",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return _ProbeResult(
                name="kokoro",
                ok=True,
                detail=f"weights cached at {candidate}",
            )

    return _ProbeResult(
        name="kokoro",
        ok=False,
        detail=(
            f"no Kokoro weights found under {project_cache} or "
            f"{home / '.cache' / 'huggingface'}. They auto-download on first "
            f"run; install with `pip install \"booktoanime[kokoro]\"` if "
            f"missing."
        ),
    )


def _probe_binary(binary: str) -> _ProbeResult:
    resolved = shutil.which(binary)
    if resolved is None:
        return _ProbeResult(
            name=binary,
            ok=False,
            detail=f"{binary} not on PATH",
        )
    return _ProbeResult(
        name=binary,
        ok=True,
        detail=f"found at {resolved}",
    )


def _run_preflight(config: Mapping[str, Any], data_dir: Path) -> list[_ProbeResult]:
    return [
        _probe_ollama(config),
        _probe_kokoro(data_dir),
        _probe_binary("ffmpeg"),
        _probe_binary("tesseract"),
    ]


def _render_preflight_table(results: list[_ProbeResult]) -> None:
    table = Table(title="Preflight checks", show_lines=False)
    table.add_column("Dependency", style="bold")
    table.add_column("Status")
    table.add_column("Detail", style="dim")
    for result in results:
        status_text = "[green]OK[/green]" if result.ok else "[red]FAIL[/red]"
        table.add_row(result.name, status_text, result.detail)
    _console.print(table)


# --------------------------------------------------------------------------- run


@app.callback()
def _root(
    ctx: typer.Context,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            help="Override the job/state directory (default: platformdirs user_data_dir).",
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config", "-c",
            help="Path to config.yaml (default: ./config.yaml).",
        ),
    ] = None,
    env_file: Annotated[
        Path | None,
        typer.Option(
            "--env-file",
            help="Path to a .env file (default: ./.env). Loaded before config.yaml so api_key_env entries can resolve.",
        ),
    ] = None,
) -> None:
    # Load .env first — config.yaml's `api_key_env: NAME` entries are read
    # at provider-build time and rely on os.environ being populated.
    dotenv_path = (env_file or Path.cwd() / ".env").expanduser()
    load_dotenv(dotenv_path)
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = _resolve_data_dir(data_dir)
    ctx.obj["config_path"] = config


@app.command(help="Start the local FastAPI server and open a browser tab.")
def run(
    ctx: typer.Context,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    skip_preflight: Annotated[
        bool,
        typer.Option(
            "--skip-preflight",
            help="Skip the dependency probes (Ollama, Kokoro, ffmpeg, tesseract).",
        ),
    ] = False,
) -> None:
    config = _load_yaml_config(ctx.obj["config_path"])
    settings = _build_settings(config, ctx.obj["data_dir"])

    if not skip_preflight:
        results = _run_preflight(config, ctx.obj["data_dir"])
        _render_preflight_table(results)
        if any(not r.ok for r in results):
            _console.print(
                "[bold red]Preflight failed.[/bold red] Fix the items above "
                "or pass [bold]--skip-preflight[/bold] to bypass."
            )
            raise typer.Exit(code=1)

    fastapi_app = create_app(settings)

    if open_browser:
        url = f"http://{host}:{port}/"
        threading.Timer(1.0, lambda: webbrowser.open_new_tab(url)).start()

    config_uvicorn = uvicorn.Config(
        fastapi_app,
        host=host,
        port=port,
        log_level="info",
        loop="asyncio",
    )
    server = uvicorn.Server(config_uvicorn)
    server.run()


@app.command(help="Resume a previously failed/cancelled job by id.")
def resume(
    ctx: typer.Context,
    job_id: str,
) -> None:
    config = _load_yaml_config(ctx.obj["config_path"])
    settings = _build_settings(config, ctx.obj["data_dir"])

    job_dir = settings.data_dir / "jobs" / job_id
    manifest_path = job_dir / "manifest.json"
    if not manifest_path.is_file():
        _console.print(f"[red]No manifest at {manifest_path}[/red]")
        raise typer.Exit(code=1)

    from .api.deps import JobRunner
    from .state import JobRepository, open_database

    connection = open_database(settings.data_dir / "state.db")
    repo = JobRepository(connection)
    runner = JobRunner(
        data_dir=settings.data_dir,
        repo=repo,
        provider_factory=settings.provider_factory,
    )

    async def _run() -> None:
        # Touch the manifest just to fail fast on a corrupt or stale-schema
        # one before launch.
        JobManifest.from_path(manifest_path)
        running = runner.start(job_id=job_id)
        try:
            await running.task
        except BookToAnimeError as exc:
            _console.print(f"[red]{exc.user_message}[/red]")
            raise typer.Exit(code=1) from exc
        finally:
            await runner.shutdown()

    asyncio.run(_run())


@app.command(help="Probe the free-stack dependencies and exit non-zero on any failure.")
def check(ctx: typer.Context) -> None:
    config = _load_yaml_config(ctx.obj["config_path"])
    data_dir: Path = ctx.obj["data_dir"]
    results = _run_preflight(config, data_dir)
    _render_preflight_table(results)
    if any(not r.ok for r in results):
        raise typer.Exit(code=1)


@app.command(help="Print the package version and exit.")
def version() -> None:
    _console.print(f"booktoanime {__version__}")


def main() -> None:
    try:
        app()
    except BookToAnimeError as exc:
        _console.print(f"[red]{exc.user_message}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
