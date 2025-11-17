import pytest
import pandas as pd
import numpy as np
from P07_fuzzy_dedup.features.base import BasePairFeaturizer


class DummyFeaturizer(BasePairFeaturizer):
    """Minimal concrete subclass for testing BasePairFeaturizer behavior."""
    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        # Simple: return length of each string in left_col
        lengths = X[self.left_col].str.len().to_numpy().reshape(-1, 1)
        return lengths

    def get_feature_names_out(self, input_features=None):
        return np.array(["dummy_length"])


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "left_name": ["alpha", "beta"],
        "right_name": ["gamma", "delta"]
    })


def test_can_instantiate_and_fit(sample_df):
    feat = DummyFeaturizer()
    fitted = feat.fit(sample_df)
    assert hasattr(fitted, "fitted_")
    assert fitted.fitted_ is True

def test_transform_returns_correct_shape(sample_df):
    feat = DummyFeaturizer().fit(sample_df)
    out = feat.transform(sample_df)
    assert out.shape == (len(sample_df), 1)
    assert out[0, 0] == len(sample_df["left_name"][0])

def test_get_feature_names_out():
    feat = DummyFeaturizer()
    names = feat.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names[0] == "dummy_length"

def test_context_methods_fallback(sample_df):
    feat = DummyFeaturizer().fit_with_context(sample_df, context={"dummy": True})
    assert hasattr(feat, "fitted_")
    out = feat.transform_with_context(sample_df, context={"dummy": True})
    assert out.shape[0] == len(sample_df)
