#!/usr/bin/python3

"""
This module contains unit tests for the DataProcessor class.
"""
import numpy as np
import unittest
import pandas as pd
from src.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """
    Unit tests for the DataProcessor class.
    """

    def setUp(self):
        """
        Initializes test fixtures before each test method.

        Creates a sample DataFrame and initializes a DataProcessor instance.
        """
        data = {
            'AccountId': [1, 1, 2, 2],
            'Amount': [100, 200, 150, 250],
            'TransactionStartTime': pd.date_range(
                start='1/1/2022', periods=4, freq='M')
        }
        # self.df = pd.DataFrame(data)
        self.df = pd.read_csv('data/test.csv')
        self.processor = DataProcessor(self.df)

    def test_calculate_aggregate_features(self):
        """
        Tests the calculation of aggregate features.

        Asserts that the aggregate features are correctly
        added to the DataFrame.
        """
        self.processor.calculate_aggregate_features()
        self.assertIn('TotalTransactionAmount', self.processor.df.columns)
        self.assertIn('AverageTransactionAmount', self.processor.df.columns)

    def test_extract_time_features(self):
        """
        Tests the extraction of time-based features.

        Asserts that the time-based features are
        correctly added to the DataFrame.
        """
        self.processor.extract_time_features()
        self.assertIn('TransactionHour', self.processor.df.columns)
        self.assertIn('TransactionDay', self.processor.df.columns)

    def test_encode_categorical_features(self):
        """
        Tests the encoding of categorical features.

        Asserts that the categorical features are correctly encoded.
        """
        self.processor.encode_categorical_features()
        self.assertIn('AccountId_LabelEnc', self.processor.df.columns)

    def test_normalize_features(self):
        """
        Tests normalization of numerical features.

        Asserts that numerical features are correctly normalized.
        """
        self.processor.calculate_aggregate_features()
        self.processor.extract_time_features()
        self.processor.encode_categorical_features()

        self.processor.normalize_features()
        self.assertTrue(self.processor.df['TotalTransactionAmount'].std() != 0)

    def test_preprocess(self):
        """
        Tests the full preprocessing pipeline.

        Asserts that the DataFrame is correctly preprocessed.
        """
        processed_df = self.processor.preprocess()
        self.assertIn('TotalTransactionAmount', processed_df.columns)
        self.assertIn('TransactionHour', processed_df.columns)
        self.assertIn('AccountId_LabelEnc', processed_df.columns)


if __name__ == '__main__':
    unittest.main()
