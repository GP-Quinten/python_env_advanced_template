import numpy as np
import pandas as pd
from P07_fuzzy_dedup.features.lcs_featurizer import LCSNormalizedFeaturizer

def test_lcs_featurizer_transform():
    df = pd.DataFrame({"left_name": ["abcdef"], "right_name": ["acf"]})
    feat = LCSNormalizedFeaturizer()
    feat.fit(df)
    out = feat.transform(df)
    assert out.shape == (1, 1)
    assert 0.0 <= out[0, 0] <= 1.0

def test_lcs_feature_names():
    feat = LCSNormalizedFeaturizer()
    names = feat.get_feature_names_out()
    assert np.array_equal(names, ["lcs_norm"])
