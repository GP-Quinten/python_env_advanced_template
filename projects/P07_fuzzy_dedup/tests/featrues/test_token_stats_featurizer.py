import pandas as pd
import numpy as np
from P07_fuzzy_dedup.features.token_stats_featurizer import TokenStatsFeaturizer

def test_token_stats_basic():
    df = pd.DataFrame({
        "left_name": ["a b c", "a b"],
        "right_name": ["a b c", "x y z"]
    })
    feat = TokenStatsFeaturizer(stopwords=None)
    feat.fit(df)
    X = feat.transform(df)
    assert X.shape == (2, 3)
    # identical sets -> overlap_coef=1, diff=0, common_frac=1
    assert np.allclose(X[0], [1.0, 0.0, 1.0])
    # no common tokens -> overlap=0, diff=1, common_frac=0
    assert X[1, 0] == 0.0
    assert X[1, 2] == 0.0
