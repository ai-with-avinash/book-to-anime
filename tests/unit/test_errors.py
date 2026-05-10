"""Smoke tests for error type contract."""

from __future__ import annotations

from booktoanime.errors import (
    BookToAnimeError,
    CapabilityNotSupported,
    CorruptedPDFError,
    EncryptedPDFError,
    OCRUnavailableError,
    ParsingError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTransientError,
    UnparseableImageOnlyPDFError,
)


def test_parsing_errors_are_book_to_anime_errors() -> None:
    for cls in (
        ParsingError,
        EncryptedPDFError,
        CorruptedPDFError,
        UnparseableImageOnlyPDFError,
        OCRUnavailableError,
    ):
        assert issubclass(cls, BookToAnimeError)


def test_provider_errors_are_book_to_anime_errors() -> None:
    for cls in (
        ProviderError,
        ProviderAuthError,
        ProviderRateLimitError,
        ProviderTransientError,
        CapabilityNotSupported,
    ):
        assert issubclass(cls, BookToAnimeError)


def test_user_message_override() -> None:
    err = EncryptedPDFError(user_message="custom message")
    assert err.user_message == "custom message"
    assert "custom message" in str(err)


def test_default_user_message_present() -> None:
    err = EncryptedPDFError()
    assert "encrypted" in err.user_message.lower()
