import numpy as np
import pandas as pd
from P07_fuzzy_dedup.features.jw_featurizer import JaroWinklerFeaturizer

def test_jw_featurizer_transform():
    df = pd.DataFrame({"left_name": ["martha"], "right_name": ["marhta"]})
    feat = JaroWinklerFeaturizer()
    feat.fit(df)
    out = feat.transform(df)
    assert out.shape == (1, 1)
    assert 0.0 <= out[0, 0] <= 1.0

def test_jw_feature_names():
    feat = JaroWinklerFeaturizer()
    names = feat.get_feature_names_out()
    assert np.array_equal(names, ["jw"])
