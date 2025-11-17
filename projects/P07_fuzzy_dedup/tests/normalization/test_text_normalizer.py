import pytest
import pandas as pd
from P07_fuzzy_dedup.normalization.text_normalizer import RegistryNameNormalizer

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "left_name": ["The Centre for Prog Study", "Registry Alpha"],
        "right_name": ["Centre Prog", "Registry Beta"]
    })

def test_fit_and_transform_replace_mode(sample_df):
    norm = RegistryNameNormalizer(output_mode="replace")
    norm.fit(sample_df)
    transformed = norm.transform(sample_df)
    assert list(transformed.columns) == ["left_name", "right_name"]
    assert transformed.shape[0] == sample_df.shape[0]
    assert all(isinstance(val, str) for val in transformed["left_name"])

def test_fit_and_transform_append_mode(sample_df):
    norm = RegistryNameNormalizer(output_mode="append", normalized_suffix="_normalized")
    norm.fit(sample_df)
    transformed = norm.transform(sample_df)
    # Original + 2 normalized columns
    assert transformed.shape[1] == sample_df.shape[1] + 2
    assert any(col.endswith("_normalized") for col in transformed.columns)

def test_custom_stopwords_and_drop_terms(sample_df):
    custom_drop_terms = {"alpha"}
    norm = RegistryNameNormalizer(
        drop_terms=custom_drop_terms,
        stopwords=set(),
        output_mode="replace"
    )
    norm.fit(sample_df)
    transformed = norm.transform(sample_df)
    # 'alpha' should be dropped
    assert all("alpha" not in val for val in transformed["left_name"])

def test_missing_columns_raises():
    df = pd.DataFrame({"col_a": ["a"], "col_b": ["b"]})
    norm = RegistryNameNormalizer()
    with pytest.raises(KeyError):
        norm.fit(df)

def test_non_dataframe_input_raises():
    norm = RegistryNameNormalizer()
    with pytest.raises(TypeError):
        norm.fit([["a", "b"]])
