"""Replicate-hosted SadTalker lip-sync adapter.

For users without a CUDA box (notably Macs), Replicate runs SadTalker as a
serverless model at a fixed per-second cost. We POST the image + audio,
poll the prediction endpoint, then stream the resulting mp4 to disk.

API contract followed:
    https://replicate.com/cjwbw/sadtalker/api

The model id is configurable so users can switch to a different SadTalker
fork (e.g. ``lucataco/sadtalker``) without code changes.
"""

from __future__ import annotations

import asyncio
import base64
import os
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx

from ...errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from ..base import AnimatedShot, LipSyncProvider
from ..registry import register_lipsync_provider

_DEFAULT_API_BASE = "https://api.replicate.com/v1"
_DEFAULT_MODEL = "cjwbw/sadtalker"
_POLL_INTERVAL_SECONDS = 2.0
_MAX_POLL_SECONDS = 300.0


class ReplicateHostedProvider(LipSyncProvider):
    name = "replicate"

    def __init__(
        self,
        *,
        api_token: str,
        model: str = _DEFAULT_MODEL,
        version: str | None = None,
        api_base: str = _DEFAULT_API_BASE,
        timeout_seconds: float = 60.0,
        still_mode: bool = True,
        preprocess: str = "full",
    ) -> None:
        if not api_token:
            raise ValueError("ReplicateHostedProvider requires a non-empty api_token")
        self._token = api_token
        self._model = model
        self._version = version
        self._api_base = api_base.rstrip("/")
        self._still_mode = still_mode
        self._preprocess = preprocess
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 10.0)),
            headers={"Authorization": f"Token {api_token}", "Content-Type": "application/json"},
        )

    async def animate(
        self,
        *,
        image_path: Path,
        audio_path: Path,
        out_path: Path,
    ) -> AnimatedShot:
        if not image_path.is_file():
            raise ProviderError(f"persona image missing: {image_path}")
        if not audio_path.is_file():
            raise ProviderError(f"shot audio missing: {audio_path}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

        prediction = await self._create_prediction(image_path, audio_path)
        prediction_id = prediction["id"]
        result = await self._poll_until_complete(prediction_id)
        output_url = _extract_output_url(result)
        await self._stream_to(output_url, tmp_path)

        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            with suppress(FileNotFoundError):
                tmp_path.unlink()
            raise ProviderError("replicate returned an empty mp4")
        tmp_path.replace(out_path)

        return AnimatedShot(
            path=out_path,
            duration_seconds=0.0,  # MouthAnimator re-measures with ffprobe.
            fps=25.0,
        )

    async def close(self) -> None:
        await self._client.aclose()

    # --------------------------------------------------------- internals

    async def _create_prediction(
        self,
        image_path: Path,
        audio_path: Path,
    ) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "input": {
                "source_image": _to_data_url(image_path, "image/png"),
                "driven_audio": _to_data_url(audio_path, "audio/wav"),
                "still": self._still_mode,
                "preprocess": self._preprocess,
            },
        }
        if self._version is not None:
            payload["version"] = self._version
            url = f"{self._api_base}/predictions"
        else:
            url = f"{self._api_base}/models/{self._model}/predictions"
        response = await self._client.post(url, json=payload)
        _raise_for_status(response)
        body: Mapping[str, Any] = response.json()
        return body

    async def _poll_until_complete(self, prediction_id: str) -> Mapping[str, Any]:
        url = f"{self._api_base}/predictions/{prediction_id}"
        elapsed = 0.0
        while elapsed < _MAX_POLL_SECONDS:
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            elapsed += _POLL_INTERVAL_SECONDS
            response = await self._client.get(url)
            _raise_for_status(response)
            body: Mapping[str, Any] = response.json()
            status = str(body.get("status"))
            if status == "succeeded":
                return body
            if status in {"failed", "canceled"}:
                detail = str(body.get("error") or "unknown failure")
                raise ProviderError(f"replicate prediction {status}: {detail}")
        raise ProviderTransientError(
            f"replicate prediction {prediction_id!r} did not finish within "
            f"{_MAX_POLL_SECONDS:.0f}s"
        )

    async def _stream_to(self, url: str, dest: Path) -> None:
        async with self._client.stream("GET", url) as response:
            _raise_for_status(response)
            with dest.open("wb") as handle:
                async for chunk in response.aiter_bytes():
                    handle.write(chunk)


def _to_data_url(path: Path, mime: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_output_url(result: Mapping[str, Any]) -> str:
    output = result.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, str):
            return first
    raise ProviderError(f"replicate response missing 'output' url: {result!r}")


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    if response.status_code in {401, 403}:
        raise ProviderAuthError("replicate rejected the api token (401/403)")
    if response.status_code == 429:
        raise ProviderRateLimitError("replicate rate limited (429)")
    if response.status_code >= 500:
        raise ProviderTransientError(f"replicate {response.status_code} server error")
    raise ProviderError(
        f"replicate {response.status_code}: {response.text[:200]}"
    )


@register_lipsync_provider("replicate")
def _factory(sub_config: Mapping[str, Any]) -> ReplicateHostedProvider:
    token_env = str(sub_config.get("api_key_env", "REPLICATE_API_TOKEN"))
    token = sub_config.get("api_key") or os.environ.get(token_env)
    if not token:
        raise ProviderAuthError(
            f"replicate lipsync requires {token_env} env var or "
            "lipsync.replicate.api_key in config"
        )
    return ReplicateHostedProvider(
        api_token=str(token),
        model=str(sub_config.get("model", _DEFAULT_MODEL)),
        version=(str(sub_config["version"]) if "version" in sub_config else None),
        api_base=str(sub_config.get("api_base", _DEFAULT_API_BASE)),
        timeout_seconds=float(sub_config.get("request_timeout_s", 60)),
        still_mode=bool(sub_config.get("still_mode", True)),
        preprocess=str(sub_config.get("preprocess", "full")),
    )


__all__ = ["ReplicateHostedProvider"]
