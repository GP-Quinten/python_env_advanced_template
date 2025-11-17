import numpy as np
import pandas as pd
from P07_fuzzy_dedup.features.levenshtein_featurizer import LevenshteinNormSimFeaturizer

def test_levenshtein_featurizer_transform():
    df = pd.DataFrame({"left_name": ["kitten"], "right_name": ["sitting"]})
    feat = LevenshteinNormSimFeaturizer()
    feat.fit(df)
    out = feat.transform(df)
    assert out.shape == (1, 1)
    assert 0.0 <= out[0, 0] <= 1.0

def test_levenshtein_feature_names():
    feat = LevenshteinNormSimFeaturizer()
    names = feat.get_feature_names_out()
    assert np.array_equal(names, ["lev_sim"])
