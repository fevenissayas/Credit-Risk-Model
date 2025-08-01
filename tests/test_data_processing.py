import pytest
import pandas as pd
import numpy as np
from src.data_processing import DateTimeFeatureExtractor, ColumnDropper, CustomerAggregator


def test_datetime_feature_extractor_columns():
    """
    Test that DateTimeFeatureExtractor correctly adds new time-based columns
    and removes the original datetime column.
    """
    # dummy DataFrame
    data = {
        'TransactionStartTime': ['2023-01-01 10:30:00', '2023-02-15 22:05:10', '2023-07-20 03:00:00'],
        'OtherFeature': [1, 2, 3]
    }
    df = pd.DataFrame(data)

    extractor = DateTimeFeatureExtractor(date_column='TransactionStartTime')
    transformed_df = extractor.transform(df)

    # Assert that the original column is dropped
    assert 'TransactionStartTime' not in transformed_df.columns

    # Assert that new columns are added
    assert 'TransactionHour' in transformed_df.columns
    assert 'TransactionDay' in transformed_df.columns
    assert 'TransactionMonth' in transformed_df.columns
    assert 'TransactionYear' in transformed_df.columns

    # Assert correct data types for new columns (should be numerical)
    assert pd.api.types.is_numeric_dtype(transformed_df['TransactionHour'])
    assert pd.api.types.is_numeric_dtype(transformed_df['TransactionDay'])
    assert pd.api.types.is_numeric_dtype(transformed_df['TransactionMonth'])
    assert pd.api.types.is_numeric_dtype(transformed_df['TransactionYear'])


def test_datetime_feature_extractor_values():
    # Test that DateTimeFeatureExtractor extracts correct values.
    data = {
        'TransactionStartTime': ['2023-01-01 10:30:00', '2022-12-25 23:59:00'],
        'OtherFeature': [1, 2]
    }
    df = pd.DataFrame(data)

    extractor = DateTimeFeatureExtractor(date_column='TransactionStartTime')
    transformed_df = extractor.transform(df)

    # Check extracted values for the first row
    assert transformed_df['TransactionHour'].iloc[0] == 10
    assert transformed_df['TransactionDay'].iloc[0] == 1
    assert transformed_df['TransactionMonth'].iloc[0] == 1
    assert transformed_df['TransactionYear'].iloc[0] == 2023

    # Check extracted values for the second row
    assert transformed_df['TransactionHour'].iloc[1] == 23
    assert transformed_df['TransactionDay'].iloc[1] == 25
    assert transformed_df['TransactionMonth'].iloc[1] == 12
    assert transformed_df['TransactionYear'].iloc[1] == 2022

def test_datetime_feature_extractor_missing_column():
    data = {
        'SomeOtherColumn': [1, 2, 3]
    }
    df = pd.DataFrame(data)

    extractor = DateTimeFeatureExtractor(date_column='NonExistentColumn')
    transformed_df = extractor.transform(df)

    # Assert that the DataFrame remains unchanged (no new columns, no errors)
    assert list(transformed_df.columns) == list(df.columns)
    assert transformed_df.shape == df.shape

def test_datetime_feature_extractor_invalid_dates():
    data = {
        'TransactionStartTime': ['2023-01-01 10:00:00', 'invalid-date-string', '2023-03-01 12:00:00'],
        'OtherFeature': [1, 2, 3]
    }
    df = pd.DataFrame(data)

    extractor = DateTimeFeatureExtractor(date_column='TransactionStartTime')
    transformed_df = extractor.transform(df)

    assert np.isnan(transformed_df['TransactionHour'].iloc[1])
    assert np.isnan(transformed_df['TransactionDay'].iloc[1])
    assert np.isnan(transformed_df['TransactionMonth'].iloc[1])
    assert np.isnan(transformed_df['TransactionYear'].iloc[1])

    assert transformed_df['TransactionHour'].iloc[0] == 10
    assert transformed_df['TransactionHour'].iloc[2] == 12


# --- Test for ColumnDropper ---

def test_column_dropper_single_column():
    # Test that ColumnDropper correctly drops a single specified column.
    data = {'A': [1, 2], 'B': [3, 4], 'C': [5, 6]}
    df = pd.DataFrame(data)
    dropper = ColumnDropper(columns_to_drop=['B'])
    transformed_df = dropper.transform(df)
    assert 'B' not in transformed_df.columns
    assert list(transformed_df.columns) == ['A', 'C']

def test_column_dropper_multiple_columns():
    # Test that ColumnDropper correctly drops multiple specified columns.
    data = {'A': [1, 2], 'B': [3, 4], 'C': [5, 6]}
    df = pd.DataFrame(data)
    dropper = ColumnDropper(columns_to_drop=['A', 'C'])
    transformed_df = dropper.transform(df)
    assert 'A' not in transformed_df.columns
    assert 'C' not in transformed_df.columns
    assert list(transformed_df.columns) == ['B']

def test_column_dropper_non_existent_column():
    # Test that ColumnDropper does not raise an error if a column to drop does not exist.
    data = {'A': [1, 2], 'B': [3, 4]}
    df = pd.DataFrame(data)
    dropper = ColumnDropper(columns_to_drop=['C', 'D']) # C and D do not exist
    transformed_df = dropper.transform(df)
    assert list(transformed_df.columns) == list(df.columns)
    assert transformed_df.shape == df.shape