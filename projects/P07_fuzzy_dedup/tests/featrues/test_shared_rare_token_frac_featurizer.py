import pandas as pd
import numpy as np
from P07_fuzzy_dedup.features.shared_rare_token_frac_featurizer import SharedRareTokenFracFeaturizer

def test_shared_rare_token_frac_basic():
    df = pd.DataFrame({
        "left_name": ["rare common", "apple banana"],
        "right_name": ["rare common", "apple pear"]
    })
    feat = SharedRareTokenFracFeaturizer(percentile=0)  # low threshold -> all tokens "rare"
    feat.fit(df)
    X = feat.transform(df)
    assert X.shape == (2, 1)
    # identical with all tokens rare -> fraction=1
    assert np.isclose(X[0, 0], 1.0)
    # pair with some shared rare tokens
    assert 0 <= X[1, 0] <= 1
