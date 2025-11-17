import math
import mmh3  # MurmurHash3 hashing library
from itertools import combinations
from collections import defaultdict
from typing import Iterable, Iterator, Tuple, Set, Dict, List, Optional, Union

DocId = Union[str, int]
BucketId = str


class LSHBucketer:
    """
    LSHBucketer: Group similar strings into "buckets" using MinHash + LSH.

    This class is used for:
      - **Batch indexing** (ML/data cleaning): Build an index from many strings and
        then find all possible candidate pairs that are likely similar.
      - **Production matching** (real-time): For each new string, quickly find
        existing strings that are probably similar, then optionally insert the new one.

    **Key concepts:**
      - **n-gram**: Break a string into overlapping chunks of length n (e.g., "cat" → "ca", "at").
      - **Shingle hashing**: Convert each n-gram to an integer using a fast hash function.
      - **MinHash**: Approximate Jaccard similarity by taking the smallest hash value
        across shingles for each of many random hash functions.
      - **Bands**: Split the MinHash signature into groups; if two signatures are
        identical in *any* band, they are considered a "match".
      - **Buckets**: Each band hashes to a bucket ID; similar strings fall into the same bucket.

    **Example (quick demo):**
    ```python
    from lsh_bucketer import LSHBucketer

    # Create LSH bucketer with small parameters for testing
    bucketer = LSHBucketer(ngram=3, num_perm=64, bands=16, seed=42)

    # Build index from a few items (doc_id, text)
    bucketer.build([
        (1, "apple juice"),
        (2, "apple juce"),  # typo: "juice" → "juce"
        (3, "orange juice")
    ])

    # Find all candidate pairs (likely similar)
    print(list(bucketer.pairs()))
    # Might output: [(1, 2), (1, 3)] depending on parameters

    # Production-style query (without inserting)
    print(bucketer.query("apple juic"))
    # Might output: {1, 2}

    # Production-style upsert (query + insert)
    print(bucketer.upsert(4, "appl juice"))
    # Returns matches found before inserting 4
    ```
    """

    def __init__(
        self,
        *,
        ngram: int = 3,
        num_perm: int = 128,
        bands: int = 32,
        seed: int = 13,
        min_block_size: int = 2,
        max_block_size: int = 200,
    ) -> None:
        """
        Create an empty LSH index with the given parameters.

        Args:
            ngram: Size of character n-grams (default 3 = trigrams).
                   Example: "cat" → ["^^c", "^ca", "cat", "at$", "t$$"] for n=3.
            num_perm: Length of the MinHash signature (number of hash functions).
            bands: Number of bands to split the signature into (LSH sensitivity).
            seed: Random seed for reproducibility (controls hash functions).
            min_block_size: Ignore buckets smaller than this size (too few to be useful).
            max_block_size: Ignore buckets larger than this size (too common, noisy).

        Note:
            - **Lower bands** → higher threshold (fewer matches, more precision).
            - **More bands** → lower threshold (more matches, more recall).
        """
        self.ngram = ngram
        self.num_perm = num_perm
        self.bands = bands
        self.seed = seed
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size

        # Large prime for modular hashing in MinHash
        self._P = (1 << 61) - 1

        # Pre-generate hash function parameters for MinHash
        self._hash_funcs = self._make_hash_funcs(num_perm, seed)

        # Internal storage
        self._bucket_to_ids: Dict[BucketId, Set[DocId]] = defaultdict(set)
        self._id_to_buckets: Dict[DocId, List[BucketId]] = {}

    # ----------------- Persistence -----------------

    @classmethod
    def load(cls, path: str) -> "LSHBucketer":
        """
        Load an LSH index from a file created with `save()`.

        Args:
            path: Path to file saved by `save()`.

        Returns:
            LSHBucketer instance with index and config restored.
        """
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise ValueError(f"File {path} is not an LSHBucketer")
        return obj

    def save(self, path: str) -> None:
        """
        Save the current LSH index and configuration to a file.

        Args:
            path: Destination file path.
        """
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    # ----------------- Index building -----------------

    def build(self, docs: Iterable[Tuple[DocId, str]]) -> None:
        """
        Build the index from scratch.

        Args:
            docs: Iterable of (doc_id, text) tuples.

        Note:
            Clears any existing index before adding docs.
        """
        self._bucket_to_ids.clear()
        self._id_to_buckets.clear()
        self.add_many(docs)

    def add_many(self, docs: Iterable[Tuple[DocId, str]]) -> int:
        """
        Add or update multiple documents to the index.

        Args:
            docs: Iterable of (doc_id, text) tuples.

        Returns:
            The number of documents processed.
        """
        count = 0
        for doc_id, text in docs:
            self._index_one(doc_id, text)
            count += 1
        return count

    # ----------------- Querying -----------------

    def query(self, text: str, *, limit: Optional[int] = None) -> Set[DocId]:
        """
        Find candidates that share at least one LSH bucket with the given text.

        Args:
            text: The string to search for.
            limit: Optional cap on number of results.

        Returns:
            A set of DocIds for similar items (empty set if none).
        """
        buckets = self._blocks_for_text(text)
        candidates: Set[DocId] = set()
        for b in buckets:
            ids = self._bucket_to_ids.get(b, ())
            if self.min_block_size <= len(ids) <= self.max_block_size:
                candidates.update(ids)
        if limit is not None and len(candidates) > limit:
            return set(list(candidates)[:limit])
        return candidates

    def upsert(self, doc_id: DocId, text: str, *, limit: Optional[int] = None) -> Set[DocId]:
        """
        Find candidates for the text, then insert/update it in the index.

        Args:
            doc_id: Unique identifier for the text.
            text: The string to insert/update.
            limit: Optional cap on number of returned matches.

        Returns:
            A set of DocIds for similar items found before insertion.
        """
        candidates = self.query(text, limit=limit)
        self._remove(doc_id)
        self._index_one(doc_id, text)
        return candidates

    def pairs(self) -> Iterator[Tuple[DocId, DocId]]:
        """
        Yield all unique candidate pairs (doc_id_i, doc_id_j) with i < j.

        Note:
            This is intended for ML training or bulk deduplication.

        Example:
            ```python
            for i, j in bucketer.pairs():
                print(i, j)
            ```
        """
        seen_pairs: Set[Tuple[DocId, DocId]] = set()
        for bucket_id, ids in self._bucket_to_ids.items():
            if self.min_block_size <= len(ids) <= self.max_block_size:
                sorted_ids = sorted(ids)
                for i, j in combinations(sorted_ids, 2):
                    if (i, j) not in seen_pairs:
                        seen_pairs.add((i, j))
                        yield (i, j)

    def remove(self, doc_id: DocId) -> bool:
        """
        Remove a document from the index.

        Returns:
            True if removed, False if not found.
        """
        return self._remove(doc_id)

    def stats(self) -> Dict[str, object]:
        """
        Get basic statistics about the index.

        Returns:
            Dictionary with counts and bucket size summary.
        """
        num_docs = len(self._id_to_buckets)
        num_buckets = len(self._bucket_to_ids)
        bucket_sizes = [len(ids) for ids in self._bucket_to_ids.values()]
        avg_size = sum(bucket_sizes) / len(bucket_sizes) if bucket_sizes else 0
        median_size = 0
        if bucket_sizes:
            sorted_sizes = sorted(bucket_sizes)
            mid = len(sorted_sizes) // 2
            median_size = sorted_sizes[mid] if len(sorted_sizes) % 2 else \
                (sorted_sizes[mid - 1] + sorted_sizes[mid]) / 2

        return {
            "num_docs": num_docs,
            "num_buckets": num_buckets,
            "avg_bucket_size": avg_size,
            "median_bucket_size": median_size,
        }

    # ----------------- Internal helpers -----------------

    def _char_ngrams(self, s: str) -> List[str]:
        """
        Generate character n-grams from a string.

        Args:
            s: Input string.

        Returns:
            A list of n-grams with padding added to the start and end of the string.

        Example:
            For ngram=3 and s="cat", the output will be:
            ["^^c", "^ca", "cat", "at$", "t$$"]
        """
        if not s:
            return []
        n = self.ngram
        pad = "^" * (n - 1)
        tail = "$" * (n - 1)
        s2 = f"{pad}{s}{tail}"
        return [s2[i:i + n] for i in range(len(s2) - n + 1)]

    def _shingle_to_int(self, g: str) -> int:
        """
        Convert a shingle (n-gram) into a 64-bit integer using MurmurHash3.

        Args:
            g: The shingle (n-gram) string.

        Returns:
            A 64-bit integer hash of the shingle.

        Example:
            For g="cat", the output might be a hash like 1234567890123456789.
        """
        h64_lo, _ = mmh3.hash64(g, seed=self.seed, signed=False)
        return h64_lo

    def _make_hash_funcs(self, num_perm: int, seed: int) -> List[Tuple[int, int]]:
        """
        Generate a list of hash functions for MinHash.

        Args:
            num_perm: Number of hash functions to generate.
            seed: Random seed for reproducibility.

        Returns:
            A list of tuples (a, b) representing hash function parameters.

        Example:
            For num_perm=2 and seed=42, the output might be:
            [(12345, 67890), (54321, 98765)]
        """
        import random
        rng = random.Random(seed)
        funcs = []
        for _ in range(num_perm):
            a = rng.randrange(1, self._P - 1)
            b = rng.randrange(0, self._P - 1)
            funcs.append((a, b))
        return funcs

    def _band_hash(self, band_idx: int, band_values: List[int]) -> str:
        """
        Compute a hash for a band of the MinHash signature.

        Args:
            band_idx: Index of the band.
            band_values: List of integers representing the band values.

        Returns:
            A string representing the hash of the band.

        Example:
            For band_idx=0 and band_values=[123, 456], the output might be:
            "abcdef1234567890"
        """
        data = bytearray()
        data += band_idx.to_bytes(4, "little", signed=False)
        for v in band_values:
            data += v.to_bytes(8, "little", signed=False)
        data += self.seed.to_bytes(8, "little", signed=False)
        h128 = mmh3.hash128(bytes(data), seed=self.seed, signed=False)
        return f"{h128:032x}"[:16]

    def _blocks_for_text(self, text: str) -> List[BucketId]:
        """
        Generate bucket IDs (blocks) for a given text.

        Args:
            text: Input string.

        Returns:
            A list of bucket IDs where the text belongs.

        Example:
            For text="apple juice", the output might be:
            ["mh3g:128x32:0:abcdef1234567890", "mh3g:128x32:1:123456abcdef7890"]
        """
        grams = self._char_ngrams(text.strip())
        if not grams:
            return []
        shingles = {self._shingle_to_int(g) for g in grams}

        sig = []
        for (a, b) in self._hash_funcs:
            m = min(((a * x + b) % self._P) for x in shingles)
            sig.append(m)

        rows = math.ceil(self.num_perm / self.bands)
        blocks: List[str] = []
        for b_idx in range(self.bands):
            start = b_idx * rows
            if start >= self.num_perm:
                break
            end = min(start + rows, self.num_perm)
            band_vals = sig[start:end]
            bucket_id = f"mh{self.ngram}g:{self.num_perm}x{self.bands}:{b_idx}:{self._band_hash(b_idx, band_vals)}"
            blocks.append(bucket_id)
        return blocks

    def _index_one(self, doc_id: DocId, text: str) -> None:
        """
        Index a single document by generating its bucket IDs and storing them.

        Args:
            doc_id: Unique identifier for the document.
            text: The text content of the document.

        Example:
            For doc_id=1 and text="apple juice", the document will be indexed
            into the corresponding buckets.
        """
        blocks = self._blocks_for_text(text)
        self._id_to_buckets[doc_id] = blocks
        for b in blocks:
            self._bucket_to_ids[b].add(doc_id)

    def _remove(self, doc_id: DocId) -> bool:
        """
        Remove a document from the index.

        Args:
            doc_id: Unique identifier for the document.

        Returns:
            True if the document was removed, False if it was not found.

        Example:
            If doc_id=1 exists in the index, it will be removed and return True.
            Otherwise, it will return False.
        """
        if doc_id not in self._id_to_buckets:
            return False
        blocks = self._id_to_buckets.pop(doc_id)
        for b in blocks:
            ids = self._bucket_to_ids.get(b)
            if ids:
                ids.discard(doc_id)
                if not ids:
                    del self._bucket_to_ids[b]
        return True
