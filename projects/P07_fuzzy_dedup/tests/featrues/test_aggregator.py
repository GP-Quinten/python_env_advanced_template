import pytest
import pandas as pd
import numpy as np
from P07_fuzzy_dedup.features.aggregator import PairFeatureUnion
from P07_fuzzy_dedup.features.base import BasePairFeaturizer


class DummyFeaturizer(BasePairFeaturizer):
    """A featurizer that outputs a single constant feature."""
    def __init__(self, value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        return np.full((len(X), 1), self.value)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"const_{self.value}"])


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "left_name": ["foo", "bar"],
        "right_name": ["baz", "qux"]
    })


def test_fit_and_transform_with_single_featurizer(sample_df):
    agg = PairFeatureUnion(featurizers=[("const", DummyFeaturizer(42))])
    agg.fit(sample_df)
    out = agg.transform(sample_df)
    assert out.shape == (len(sample_df), 1)
    assert np.all(out == 42)

def test_get_feature_names_out(sample_df):
    agg = PairFeatureUnion(featurizers=[
        ("f1", DummyFeaturizer(1)),
        ("f2", DummyFeaturizer(2))
    ])
    agg.fit(sample_df)
    names = agg.get_feature_names_out()
    assert list(names) == ["f1.const_1", "f2.const_2"]

def test_error_if_missing_columns():
    df = pd.DataFrame({"col_a": ["a"], "col_b": ["b"]})
    agg = PairFeatureUnion(featurizers=[("const", DummyFeaturizer())])
    with pytest.raises(KeyError):
        agg.fit(df)

def test_transform_without_fit_raises(sample_df):
    agg = PairFeatureUnion(featurizers=[("const", DummyFeaturizer())])
    with pytest.raises(RuntimeError):
        agg.transform(sample_df)
