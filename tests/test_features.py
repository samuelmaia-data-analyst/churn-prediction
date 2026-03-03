from __future__ import annotations

from src.data.preprocess import DataPreprocessor
from src.features.build_features import FeatureEngineer
from tests.test_data import build_raw_df


def test_feature_engineer_generates_named_dataframe() -> None:
    preprocessor = DataPreprocessor()
    data = preprocessor.clean_data(build_raw_df())
    X_train, X_test, _, _ = preprocessor.split_data(data)

    feature_engineer = FeatureEngineer()
    X_train_proc, X_test_proc = feature_engineer.fit_transform(X_train, X_test)

    assert X_train_proc.shape[1] == X_test_proc.shape[1]
    assert X_train_proc.columns.is_unique
    assert feature_engineer.feature_names is not None
    assert len(feature_engineer.feature_names) == X_train_proc.shape[1]
