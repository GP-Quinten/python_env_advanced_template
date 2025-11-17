import pandas as pd
import numpy as np
from P07_fuzzy_dedup.features.char_ngram_jaccard_featurizer import CharNgramJaccardFeaturizer

def test_char_ngram_jaccard_basic():
    df = pd.DataFrame({
        "left_name": ["abcd", "test"],
        "right_name": ["abcd", "tent"]
    })
    feat = CharNgramJaccardFeaturizer()
    feat.fit(df)
    X = feat.transform(df)
    assert X.shape == (2, 2)
    # identical strings should have jaccard=1 for both n=3 and n=4
    assert np.allclose(X[0], [1.0, 1.0])
    # partially matching strings -> jaccard between 0 and 1
    assert 0 <= X[1, 0] <= 1
    assert 0 <= X[1, 1] <= 1
