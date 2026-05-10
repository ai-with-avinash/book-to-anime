"""FastAPI app + HTML/htmx frontend.

The CLI (module 6's ``cli.py``) starts uvicorn with the app built by
:func:`create_app`. Tests can build their own app instance and inject mock
providers via the :class:`AppSettings` factory.
"""

from __future__ import annotations

from .app import AppSettings, create_app
from .deps import ProviderFactory

__all__ = ["AppSettings", "ProviderFactory", "create_app"]
