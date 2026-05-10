"""Shared helpers used by native-SDK language adapters."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


def resolve_api_key(sub_config: Mapping[str, Any], *, default_env: str) -> str:
    """Resolve an API key from explicit config or an env var.

    ``api_key_env`` falls back to ``default_env`` when omitted, matching the
    documented behavior in ``config.example.yaml``.
    """

    explicit = sub_config.get("api_key")
    if explicit:
        return str(explicit)

    env_var = str(sub_config.get("api_key_env") or default_env)
    value = os.environ.get(env_var)
    if not value:
        raise ValueError(
            f"missing API key: set ${env_var} or `api_key:` in the provider's config block."
        )
    return value


def require_sdk(import_name: str, *, install_extra: str) -> None:
    """Importable check used by adapter factories.

    The factory body has already attempted ``import <sdk>``; if it failed it
    can call this to raise a uniformly-formatted error mentioning which extra
    to install.
    """

    raise ImportError(
        f"the {import_name!r} package is required to use this provider. "
        f"Install with `pip install booktoanime[{install_extra}]`."
    )
