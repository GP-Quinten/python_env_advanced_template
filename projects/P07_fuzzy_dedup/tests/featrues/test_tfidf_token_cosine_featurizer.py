import pandas as pd
import numpy as np
from P07_fuzzy_dedup.features.tfidf_token_cosine_featurizer import TfidfTokenCosineFeaturizer

def test_tfidf_token_cosine_basic():
    df = pd.DataFrame({
        "left_name": ["apple banana", "cat dog"],
        "right_name": ["apple banana", "cat fish"]
    })
    feat = TfidfTokenCosineFeaturizer()
    feat.fit(df)
    X = feat.transform(df)
    assert X.shape == (2, 1)
    # identical pair should have similarity 1
    assert np.isclose(X[0, 0], 1.0, atol=1e-6)
    # different pair should have similarity between 0 and 1
    assert 0 <= X[1, 0] <= 1
