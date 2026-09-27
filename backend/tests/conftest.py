import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import pytest

from app.search.index import InvertedIndex


@pytest.fixture
def small_index():
    idx = InvertedIndex()
    idx.add_document(1, "operating system memory management is essential", stem=False)
    idx.add_document(2, "virtual memory extends physical memory using disk", stem=False)
    idx.add_document(3, "the cat sat on the mat", stem=False)
    idx.add_document(4, "memory memory memory management management", stem=False)
    return idx
