import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.search.snippet import generate_snippet


def test_snippet_highlights_query_terms():
    text = "This is a long document about virtual memory and how operating systems manage it."
    snippet = generate_snippet(text, "virtual memory", stem=False)
    assert "<mark>virtual</mark>" in snippet
    assert "<mark>memory</mark>" in snippet


def test_snippet_escapes_html_to_prevent_xss():
    text = "A document with <script>alert(1)</script> and the word memory in it."
    snippet = generate_snippet(text, "memory", stem=False)
    assert "<script>" not in snippet
    assert "&lt;script&gt;" in snippet


def test_snippet_on_empty_query_has_no_highlights():
    text = "Just some plain text here."
    snippet = generate_snippet(text, "", stem=False)
    assert "<mark>" not in snippet


def test_snippet_on_no_match_has_no_highlights():
    text = "Just some plain text here."
    snippet = generate_snippet(text, "nonexistentterm", stem=False)
    assert "<mark>" not in snippet
