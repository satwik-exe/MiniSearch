"""Search snippet generation.

Rather than showing the first ~200 characters of a document (which is often
unrelated to the query), we:

  1. Tokenize the original text with position spans (so we can map back to
     the raw, un-stemmed text for display).
  2. Preprocess each token the same way the index does (lowercase + stem) so
     it can be compared against the preprocessed query terms.
  3. Slide a fixed-size window over the token positions and pick the window
     containing the most distinct query-term matches (ties broken by total
     match count).
  4. Render that window's original text, wrapping matches in <mark> tags for
     highlighting (HTML-escaped first, to avoid any XSS from document
     content ending up unescaped in the response).

Example:
    Operating Systems -- Chapter 6

    "...virtual memory allows the operating system to use
    secondary storage as an extension of physical memory..."
"""
from __future__ import annotations

import html
import re
from typing import Set

from app.search.tokenizer import porter_stem, preprocess

# ASCII-only, matching tokenizer.py's word-extraction pattern (see that file
# for why re.UNICODE would be a no-op here and was removed).
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")

WINDOW_SIZE = 40  # tokens shown around the best match
SNIPPET_MAX_CHARS = 320


def _preprocess_token(tok: str, *, stem: bool) -> str:
    t = tok.lower().strip("'-")
    if stem and t:
        t = porter_stem(t)
    return t


def generate_snippet(text: str, query: str, *, stem: bool = True) -> str:
    query_terms: Set[str] = set(preprocess(query, stem=stem))
    if not query_terms or not text:
        return html.escape(text[:SNIPPET_MAX_CHARS]) + ("..." if len(text) > SNIPPET_MAX_CHARS else "")

    matches = list(_WORD_RE.finditer(text))
    if not matches:
        return html.escape(text[:SNIPPET_MAX_CHARS])

    # For each token, is its processed form a query term?
    is_match = [
        _preprocess_token(m.group(), stem=stem) in query_terms for m in matches
    ]

    # Slide a window over token indices, find the window with the most matches.
    n = len(matches)
    window = min(WINDOW_SIZE, n)
    best_start, best_score = 0, -1
    running = sum(is_match[:window])
    best_score, best_start = running, 0
    for start in range(1, n - window + 1):
        running += is_match[start + window - 1] - is_match[start - 1]
        if running > best_score:
            best_score, best_start = running, start

    best_end = best_start + window - 1
    span_start = matches[best_start].start()
    span_end = matches[best_end].end()

    # Expand slightly to whole-sentence-ish boundaries without runaway length
    prefix = "..." if span_start > 0 else ""
    suffix = "..." if span_end < len(text) else ""
    raw_snippet = text[span_start:span_end]
    if len(raw_snippet) > SNIPPET_MAX_CHARS:
        raw_snippet = raw_snippet[:SNIPPET_MAX_CHARS]
        suffix = "..."

    return prefix + _highlight(raw_snippet, query_terms, stem=stem) + suffix


def _highlight(snippet: str, query_terms: Set[str], *, stem: bool) -> str:
    """HTML-escape the snippet, then wrap matching terms in <mark> tags.
    Escaping happens BEFORE highlighting is inserted, so document content can
    never inject raw HTML (e.g. a document containing '<script>' is rendered
    as inert text, not executed) -- highlighting only adds our own safe tags
    around already-escaped text."""
    pieces = []
    last_end = 0
    for m in _WORD_RE.finditer(snippet):
        if _preprocess_token(m.group(), stem=stem) in query_terms:
            pieces.append(html.escape(snippet[last_end:m.start()]))
            pieces.append(f"<mark>{html.escape(m.group())}</mark>")
            last_end = m.end()
    pieces.append(html.escape(snippet[last_end:]))
    return "".join(pieces)
