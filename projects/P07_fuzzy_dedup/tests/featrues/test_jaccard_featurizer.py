import numpy as np
import pandas as pd
from P07_fuzzy_dedup.features.jaccard_featurizer import TokenJaccardFeaturizer

def test_jaccard_featurizer_transform():
    df = pd.DataFrame({"left_name": ["foo bar"], "right_name": ["foo baz"]})
    feat = TokenJaccardFeaturizer(stopwords=None)
    feat.fit(df)
    out = feat.transform(df)
    assert out.shape == (1, 1)
    assert 0.0 <= out[0, 0] <= 1.0

def test_jaccard_with_stopwords():
    df = pd.DataFrame({"left_name": ["the foo"], "right_name": ["foo"]})
    feat = TokenJaccardFeaturizer(stopwords={"the"})
    feat.fit(df)
    out = feat.transform(df)
    assert out[0, 0] == 1.0  # stopword removed, both have only "foo"

def test_jaccard_feature_names():
    feat = TokenJaccardFeaturizer()
    names = feat.get_feature_names_out()
    assert np.array_equal(names, ["jaccard_tok"])
