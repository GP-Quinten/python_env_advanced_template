import pytest
from P07_fuzzy_dedup.normalization import rules

def test_rules_constants_types():
    # Ensure constants exist and are correct types
    assert isinstance(rules.DEFAULT_STOPWORDS, set)
    assert isinstance(rules.DEFAULT_DROP_TERMS, set)
    assert isinstance(rules.CANONICAL_SPELLINGS, dict)
    assert isinstance(rules.EXPANSIONS, dict)

def test_rules_contents_not_empty():
    # Should have some expected keys/values
    assert "registry" in rules.DEFAULT_DROP_TERMS
    assert rules.CANONICAL_SPELLINGS.get("centre") == "center"
    assert rules.EXPANSIONS.get("reg") == "registry"

def test_rules_no_conflicting_keys():
    # A key shouldn't be both in canonical spellings and expansions (just sanity check)
    overlap = set(rules.CANONICAL_SPELLINGS) & set(rules.EXPANSIONS)
    assert not overlap, f"Overlapping keys found: {overlap}"
