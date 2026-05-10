"""Custom exceptions and a user-facing error mapper.

The orchestrator and CLI/UI layers translate these exceptions into actionable
messages for the end user (no stack traces in the UI). Full traces are written
to the per-job log file by the logging module.
"""

from __future__ import annotations


class BookToAnimeError(Exception):
    """Base class for all BookToAnime errors."""

    user_message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None, *, user_message: str | None = None) -> None:
        if user_message is not None:
            self.user_message = user_message
        super().__init__(message or self.user_message)


# ---- Parsing errors ----------------------------------------------------------


class ParsingError(BookToAnimeError):
    """Generic PDF parsing failure."""

    user_message = "Failed to parse the PDF."


class EncryptedPDFError(ParsingError):
    """PDF is password-protected."""

    user_message = (
        "This PDF is encrypted. Decrypt it (e.g. with qpdf or your PDF reader) "
        "and try again."
    )


class CorruptedPDFError(ParsingError):
    """PDF cannot be opened or its structure is invalid."""

    user_message = "This PDF appears to be corrupted and cannot be opened."


class UnparseableImageOnlyPDFError(ParsingError):
    """PDF has no extractable text layer and OCR is disabled or unavailable."""

    user_message = (
        "This PDF has no text layer (it is image-only). Enable OCR or use a different PDF."
    )


class OCRUnavailableError(ParsingError):
    """OCR was requested but Tesseract binary is missing."""

    user_message = (
        "OCR fallback was requested but the `tesseract` binary is not installed. "
        "Install Tesseract or supply a text-based PDF."
    )


# ---- Provider errors (used by later modules; defined here for one canonical place) ----


class ProviderError(BookToAnimeError):
    user_message = "A provider call failed."


class ProviderAuthError(ProviderError):
    user_message = "Provider authentication failed. Check your API key."


class ProviderRateLimitError(ProviderError):
    user_message = "Provider rate-limited the request. Retrying with backoff."


class ProviderTransientError(ProviderError):
    user_message = "A transient provider error occurred. Retrying."


class CapabilityNotSupportedError(ProviderError):
    user_message = "This provider does not support the requested capability."


# Backwards-compatible alias used in the published interface docs.
CapabilityNotSupported = CapabilityNotSupportedError
