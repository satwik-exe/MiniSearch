"""Document ingestion: turn an uploaded file into a clean text record.

Supported types: .txt, .md, .pdf

Security notes (see README "Security"):
  - filenames are sanitized (no path components, no traversal) before ever
    touching the filesystem or being stored
  - file size is capped
  - only whitelisted extensions are accepted; content-type is not trusted
    blindly since it's client-supplied
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


class UnsupportedFileType(ValueError):
    pass


class FileTooLarge(ValueError):
    pass


class CorruptFile(ValueError):
    """Raised when a file's extension is allowed but its content cannot actually
    be parsed (e.g. a renamed non-PDF file, or a truncated/corrupted PDF)."""

    pass


@dataclass
class IngestedDocument:
    filename: str
    title: str
    file_type: str
    size_bytes: int
    text: str
    word_count: int


def sanitize_filename(raw_name: str) -> str:
    """Strip any path components and unsafe characters, keeping only a safe basename.

    This prevents path traversal (e.g. "../../etc/passwd") and keeps stored/displayed
    filenames predictable.
    """
    name = Path(raw_name).name  # drop any directory components
    name = re.sub(r"[^A-Za-z0-9._\- ]", "_", name)
    name = name.strip() or "untitled"
    return name[:255]


def derive_title(filename: str) -> str:
    stem = Path(filename).stem
    title = re.sub(r"[_\-]+", " ", stem).strip()
    return title.title() if title else filename


def extract_text_from_pdf(data: bytes) -> str:
    """Extract text from a PDF using pypdf. Pages are joined with double newlines
    so downstream tokenization treats page breaks like paragraph breaks.

    An earlier version of this function let PdfReader's constructor raise
    uncaught: any corrupted PDF, truncated upload, or a non-PDF file simply
    renamed to .pdf would propagate an unhandled exception all the way up
    through ingest() -- in the FastAPI layer that becomes an unhandled 500 with
    a raw stack trace, which is exactly what "ingestion failures should return
    useful API errors rather than crashing the server" rules out. We now catch
    that specific failure and raise our own CorruptFile, which app/api/documents.py
    maps to a clean 422 response.
    """
    import io
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        len(reader.pages)  # forces parsing of the page tree now, not lazily later
    except Exception as e:
        raise CorruptFile(
            f"Could not read this PDF -- it may be corrupted, truncated, or not a valid "
            f"PDF file ({e.__class__.__name__})."
        ) from e

    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            # A single malformed page shouldn't abort ingestion of the whole document.
            pages_text.append("")
    return "\n\n".join(pages_text).strip()


def extract_text(filename: str, data: bytes) -> tuple[str, str]:
    """Return (file_type, extracted_text) for a given filename + raw bytes."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileType(
            f"'{ext}' is not supported. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    if ext in (".txt", ".md"):
        text = data.decode("utf-8", errors="replace")
    else:  # .pdf
        text = extract_text_from_pdf(data)

    file_type = ext.lstrip(".")
    return file_type, text


def ingest(raw_filename: str, data: bytes) -> IngestedDocument:
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise FileTooLarge(
            f"File exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB upload limit"
        )

    filename = sanitize_filename(raw_filename)
    file_type, text = extract_text(filename, data)
    title = derive_title(filename)
    word_count = len(text.split())

    return IngestedDocument(
        filename=filename,
        title=title,
        file_type=file_type,
        size_bytes=len(data),
        text=text,
        word_count=word_count,
    )
