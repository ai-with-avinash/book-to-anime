"""Shared pytest fixtures.

Generates synthetic PDF fixtures on demand using ``reportlab`` and Pillow so
binary files are never checked into source control. Fixtures are session-scoped
to avoid regenerating them per test.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_demo_image(text: str, *, size: tuple[int, int] = (240, 160)) -> bytes:
    """Render a small PNG with a label so it survives image extraction visibly."""

    image = Image.new("RGB", size, color=(220, 220, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, size[0] - 10, size[1] - 10], outline=(40, 40, 80), width=4)
    draw.text((24, 24), text, fill=(20, 20, 60))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_text_with_image_pdf(out_path: Path) -> None:
    """Two-page PDF with extractable text, a table-like layout, and one figure."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(out_path), pagesize=LETTER)
    pdf.setTitle("Tiny Test Book")
    pdf.setAuthor("BookToAnime Tests")

    # Page 1: text + image with a caption hint.
    pdf.setFont("Helvetica", 14)
    pdf.drawString(72, 720, "Chapter 1: Introduction")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(72, 690, "Welcome to the tiny test book. It has text and one figure.")
    pdf.drawString(72, 670, "We'll demonstrate parsing across two pages.")

    image_bytes = _make_demo_image("Figure 1.1")
    pdf.drawImage(
        ImageReader(io.BytesIO(image_bytes)),
        x=72,
        y=480,
        width=240,
        height=160,
    )
    pdf.setFont("Helvetica-Oblique", 10)
    pdf.drawString(72, 470, "Figure 1.1: Demonstration figure for parsing tests.")

    pdf.showPage()

    # Page 2: text + a small table rendered as a grid of strings.
    pdf.setFont("Helvetica", 14)
    pdf.drawString(72, 720, "Chapter 2: Data")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(72, 690, "The table below summarizes example values.")

    rows = [["Name", "Score"], ["Alice", "10"], ["Bob", "7"]]
    x0 = 72
    y0 = 640
    for row_idx, row in enumerate(rows):
        for col_idx, cell in enumerate(row):
            pdf.drawString(x0 + col_idx * 80, y0 - row_idx * 18, cell)
    pdf.showPage()

    pdf.save()


def _build_encrypted_pdf(out_path: Path) -> None:
    """Single-page PDF protected with a password reportlab handles."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(out_path), pagesize=LETTER, encrypt="hunter2")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(72, 720, "Encrypted contents.")
    pdf.showPage()
    pdf.save()


def _build_image_only_pdf(out_path: Path) -> None:
    """Single-page PDF whose only content is a rasterized image (no text layer)."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(out_path), pagesize=LETTER)
    image_bytes = _make_demo_image("(no text layer)", size=(400, 200))
    pdf.drawImage(
        ImageReader(io.BytesIO(image_bytes)),
        x=72,
        y=520,
        width=400,
        height=200,
    )
    pdf.showPage()
    pdf.save()


def _build_corrupted_pdf(out_path: Path) -> None:
    """Bytes that look like a PDF header but are otherwise malformed."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"%PDF-1.4\n%garbage that is not a real pdf\n")


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build PDF fixtures into a fresh tmp dir for the test session."""

    base = tmp_path_factory.mktemp("booktoanime-fixtures")
    _build_text_with_image_pdf(base / "tiny.pdf")
    _build_encrypted_pdf(base / "encrypted.pdf")
    _build_image_only_pdf(base / "image_only.pdf")
    _build_corrupted_pdf(base / "corrupted.pdf")
    return base


@pytest.fixture
def tiny_pdf(fixtures_dir: Path) -> Path:
    return fixtures_dir / "tiny.pdf"


@pytest.fixture
def encrypted_pdf(fixtures_dir: Path) -> Path:
    return fixtures_dir / "encrypted.pdf"


@pytest.fixture
def image_only_pdf(fixtures_dir: Path) -> Path:
    return fixtures_dir / "image_only.pdf"


@pytest.fixture
def corrupted_pdf(fixtures_dir: Path) -> Path:
    return fixtures_dir / "corrupted.pdf"


@pytest.fixture
def job_dir(tmp_path: Path) -> Path:
    """An empty per-test job directory."""

    job = tmp_path / "job"
    job.mkdir()
    return job
