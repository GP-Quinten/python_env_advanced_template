import pandas as pd
import numpy as np
from P07_fuzzy_dedup.features.tfidf_char_cosine_featurizer import TfidfCharCosineFeaturizer

def test_tfidf_char_cosine_basic():
    df = pd.DataFrame({
        "left_name": ["abcd", "apple"],
        "right_name": ["abcd", "applz"]
    })
    feat = TfidfCharCosineFeaturizer()
    feat.fit(df)
    X = feat.transform(df)
    assert X.shape == (2, 1)
    assert np.isclose(X[0, 0], 1.0, atol=1e-6)
    assert 0 <= X[1, 0] <= 1
