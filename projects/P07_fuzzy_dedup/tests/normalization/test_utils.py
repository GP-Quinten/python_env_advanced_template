import pytest
from P07_fuzzy_dedup.normalization import utils

def test_strip_accents():
    text = "Café"
    result = utils.strip_accents(text)
    assert result == "Cafe"

def test_roman_to_arabic():
    assert utils.roman_to_arabic("Type II study") == "Type 2 study"
    assert utils.roman_to_arabic("phase xiii") == "phase 13"

def test_expand_and_canonicalize():
    tokens = ["centre", "prog", "study"]
    result = utils.expand_and_canonicalize(tokens)
    # 'centre' -> 'center', 'prog' -> 'program'
    assert "center" in result
    assert "program" in result

def test_normalize_registry_names_basic():
    names = ["The Centre for Prog Study", None]
    normalized = utils.normalize_registry_names(names)
    assert isinstance(normalized, list)
    # Should lowercase, strip accents, canonicalize, drop default terms
    assert normalized[0] != names[0]
    assert all(isinstance(s, str) for s in normalized)

def test_normalize_registry_names_options():
    names = ["Registry Hospital Alpha"]
    # Keep drop terms and stopwords
    out_keep = utils.normalize_registry_names(names, remove_drop_terms=False, remove_stopwords=False)
    assert "registry" in out_keep[0]
    # Drop terms and stopwords
    out_drop = utils.normalize_registry_names(names, remove_drop_terms=True, remove_stopwords=True)
    assert "registry" not in out_drop[0]
