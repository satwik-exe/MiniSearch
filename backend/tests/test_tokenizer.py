import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.search.tokenizer import tokenize, lowercase, remove_stopwords, normalize, preprocess, porter_stem


def test_hello_world_punctuation_stripped():
    assert tokenize(lowercase("Hello, World!")) == ["hello", "world"]


def test_spec_example_machine_learning():
    text = "Machine Learning is changing the world!"
    tokens = normalize(remove_stopwords(tokenize(lowercase(text))))
    assert tokens == ["machine", "learning", "changing", "world"]


def test_stopwords_removed():
    tokens = remove_stopwords(["the", "quick", "brown", "fox", "is", "a", "fox"])
    assert "the" not in tokens
    assert "is" not in tokens
    assert "a" not in tokens
    assert tokens == ["quick", "brown", "fox", "fox"]


def test_stemming_conflates_related_forms():
    assert porter_stem("running") == porter_stem("runs") == "run"
    assert porter_stem("databases") == porter_stem("database") or porter_stem("databases").startswith("databas")


def test_empty_text_returns_empty_list():
    assert preprocess("") == []


def test_all_stopword_query_returns_empty_list():
    assert preprocess("the a an is") == []


def test_case_insensitivity():
    assert preprocess("DATABASE") == preprocess("database")


def test_punctuation_only_input():
    assert tokenize("!!! ??? ...") == []
