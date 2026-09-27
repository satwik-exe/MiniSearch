import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.documents.ingestion import ingest, sanitize_filename, UnsupportedFileType, FileTooLarge, CorruptFile


def test_txt_ingestion():
    doc = ingest("notes.txt", b"Machine Learning is changing the world!")
    assert doc.file_type == "txt"
    assert doc.word_count == 6


def test_md_ingestion():
    doc = ingest("my-notes.md", b"# Heading\n\nSome content here.")
    assert doc.file_type == "md"


def test_path_traversal_sanitized():
    assert sanitize_filename("../../etc/passwd.txt") == "passwd.txt"


def test_unsafe_characters_sanitized():
    assert "/" not in sanitize_filename("weird/name.txt")


def test_unsupported_file_type_rejected():
    with pytest.raises(UnsupportedFileType):
        ingest("archive.zip", b"PK\x03\x04")


def test_oversized_file_rejected():
    with pytest.raises(FileTooLarge):
        ingest("big.txt", b"x" * (21 * 1024 * 1024))


def test_empty_file_ingests_with_zero_words():
    doc = ingest("empty.txt", b"")
    assert doc.word_count == 0


# --- Added during audit: malformed-PDF handling ---------------------------
# An earlier version let PdfReader's constructor raise uncaught, which would
# have propagated as an unhandled 500 through the API. These confirm the fix:
# every "content doesn't match the extension" case raises a catchable,
# API-mappable CorruptFile instead of crashing.


def test_garbage_bytes_with_pdf_extension_raises_corrupt_file():
    with pytest.raises(CorruptFile):
        ingest("fake.pdf", b"this is not a real pdf file at all, just garbage bytes")


def test_empty_pdf_bytes_raises_corrupt_file():
    with pytest.raises(CorruptFile):
        ingest("empty.pdf", b"")


def test_truncated_pdf_raises_corrupt_file():
    with pytest.raises(CorruptFile):
        ingest("truncated.pdf", b"%PDF-1.4\n%truncated content with no xref table")


def test_valid_pdf_still_ingests_correctly():
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    import io

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(100, 700, "Valid PDF content for regression test.")
    c.save()
    doc = ingest("valid.pdf", buf.getvalue())
    assert doc.file_type == "pdf"
    assert doc.word_count > 0
    assert "valid pdf content" in doc.text.lower()
