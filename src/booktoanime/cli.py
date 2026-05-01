"""Top-level Typer CLI.

``booktoanime`` runs the FastAPI server. ``booktoanime resume <job_id>`` re-
runs a failed job from the last completed stage. ``booktoanime version``
prints the package version.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import webbrowser
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
import yaml
from platformdirs import user_data_dir
from rich.console import Console

from . import __version__
from .api import AppSettings, ProviderFactory, create_app
from .errors import BookToAnimeError
from .pipeline.manifest import JobManifest, ProvidersConfig

_console = Console()

app = typer.Typer(
    add_completion=False,
    help="Convert a PDF book into an anime-style narrated explainer video.",
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
        "Recommended starter paths (rough per-book hosted-API costs):\n"
        "  * [bold]Groq Llama 3.3 70B[/bold]: ~$0.05-$0.30 per book\n"
        "  * [bold]Gemini Flash[/bold]: ~$0.10-$0.50 per book\n"
        "  * [bold]Claude Sonnet[/bold]: ~$2-$8 per book\n"
        "  * [bold]GPT-4o-mini[/bold]: ~$0.50-$2 per book\n"
        "  * [bold]Local (Ollama / vLLM via OpenAI-compatible)[/bold]: $0 per book\n\n"
        "See `config.example.yaml` in the repository for the full schema."
    )


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
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = _resolve_data_dir(data_dir)
    ctx.obj["config_path"] = config


@app.command(help="Start the local FastAPI server and open a browser tab.")
def run(
    ctx: typer.Context,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    config = _load_yaml_config(ctx.obj["config_path"])
    settings = _build_settings(config, ctx.obj["data_dir"])
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
        # Touch the manifest just to fail fast on a corrupt one before launch.
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
