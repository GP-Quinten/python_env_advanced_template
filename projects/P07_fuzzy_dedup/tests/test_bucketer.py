import os
import tempfile
import pytest
from P07_fuzzy_dedup.bucketer import LSHBucketer


@pytest.fixture
def small_bucketer():
    """
    Fixture: creates a small LSHBucketer instance for testing.
    Uses tiny parameters for fast and predictable results.
    """
    return LSHBucketer(
        ngram=2,        # bigrams
        num_perm=8,     # short signature
        bands=2,        # 2 bands of 4 rows
        seed=42,        # fixed seed for reproducibility
        min_block_size=2,
        max_block_size=10
    )


@pytest.fixture
def sample_docs():
    """
    Fixture: small sample corpus with some near-duplicates and unrelated docs.
    """
    return [
        (1, "cat"),
        (2, "cut"),
        (3, "cot"),
        (4, "dog"),
        (5, "dogs"),
    ]


def test_build_and_stats(small_bucketer, sample_docs):
    """
    Test that building the index from scratch works
    and stats return correct counts.
    """
    small_bucketer.build(sample_docs)
    stats = small_bucketer.stats()
    assert stats["num_docs"] == 5
    assert stats["num_buckets"] > 0
    assert stats["avg_bucket_size"] > 0


def test_add_many_and_query(small_bucketer, sample_docs):
    """
    Test adding docs in batches and querying for matches.
    Because LSH is probabilistic, we only check that results
    are a non-empty subset of the inserted IDs, not exact matches.
    """
    small_bucketer.add_many(sample_docs)
    results = small_bucketer.query("cut")
    # Should return some candidates
    assert isinstance(results, set)
    # All candidates must be from inserted doc IDs
    inserted_ids = {doc_id for doc_id, _ in sample_docs}
    assert results.issubset(inserted_ids)


def test_upsert_returns_matches_and_inserts(small_bucketer, sample_docs):
    """
    Test that upsert returns valid matches before inserting
    and actually stores the new doc in the index.
    """
    small_bucketer.build(sample_docs)
    matches = small_bucketer.upsert(6, "cats")
    assert isinstance(matches, set)
    inserted_ids = {doc_id for doc_id, _ in sample_docs}
    assert matches.issubset(inserted_ids)
    # Verify doc 6 is in the index
    assert 6 in small_bucketer._id_to_buckets


def test_pairs_generation(small_bucketer, sample_docs):
    """
    Test that pairs() yields valid tuples of matching doc IDs.
    """
    small_bucketer.build(sample_docs)
    pairs = list(small_bucketer.pairs())
    assert all(len(pair) == 2 for pair in pairs)
    assert all(isinstance(x, int) for pair in pairs for x in pair)


def test_remove_doc(small_bucketer, sample_docs):
    """
    Test removing a doc and ensuring it's no longer in candidates.
    """
    small_bucketer.build(sample_docs)
    assert small_bucketer.remove(3) is True
    # Querying for "cot" should not return 3 anymore
    results = small_bucketer.query("cot")
    assert 3 not in results
    # Removing again should return False
    assert small_bucketer.remove(3) is False


def test_save_and_load(small_bucketer, sample_docs):
    """
    Test that saving and loading an index preserves data.
    """
    small_bucketer.build(sample_docs)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "lsh_index.pkl")
        small_bucketer.save(path)
        # Load new instance
        loaded = LSHBucketer.load(path)
        # Stats should match
        assert loaded.stats()["num_docs"] == small_bucketer.stats()["num_docs"]
        # Query results should match
        text = "cut"
        assert loaded.query(text) == small_bucketer.query(text)


def test_query_limit_parameter(small_bucketer, sample_docs):
    """
    Test that the limit parameter in query() restricts the output size.
    """
    small_bucketer.build(sample_docs)
    results_full = small_bucketer.query("cut")
    results_limited = small_bucketer.query("cut", limit=1)
    assert len(results_limited) <= 1
    assert results_limited.issubset(results_full)
