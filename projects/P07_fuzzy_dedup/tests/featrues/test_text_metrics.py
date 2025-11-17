import pytest
from P07_fuzzy_dedup.features import text_metrics as tm


def test_jaro_winkler_basic():
    assert tm.jaro_winkler("martha", "marhta") > 0.9
    assert tm.jaro_winkler("", "") == 1.0
    assert tm.jaro_winkler("abc", "") == 0.0

def test_levenshtein_distance_and_norm():
    assert tm.levenshtein_distance("kitten", "sitting") == 3
    sim = tm.levenshtein_norm_sim("kitten", "sitting")
    assert 0.5 < sim < 1.0
    assert tm.levenshtein_norm_sim("", "") == 1.0

def test_lcs_length_and_normalized():
    assert tm.lcs_length("abcdef", "acf") == 3
    assert tm.lcs_normalized("abcdef", "acf") == pytest.approx(3/6)
    assert tm.lcs_normalized("", "") == 1.0

def test_jaccard_tokens():
    s1, s2 = "foo bar", "foo baz"
    j = tm.jaccard_tokens(s1, s2)
    assert 0.0 <= j <= 1.0
    assert tm.jaccard_tokens("", "") == 1.0
    assert tm.jaccard_tokens("foo", "") == 0.0
    # With stopwords removal
    assert tm.jaccard_tokens("the foo", "foo", stopwords={"the"}) == 1.0
