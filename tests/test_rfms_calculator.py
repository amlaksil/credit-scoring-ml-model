#!/usr/bin/python3

"""
This module contains unit tests for the RFMSCalculator class.
"""

import unittest
import pandas as pd
from src.data_processor import DataProcessor
from src.rfms_calculator import RFMSCalculator


class TestRFMSCalculator(unittest.TestCase):
    """
    Unit tests for the RFMSCalculator class.
    """

    def setUp(self):
        """
        Initializes test fixtures before each test method.

        Creates a sample DataFrame and initializes an RFMSCalculator instance.
        """
        df = pd.read_csv('data/test.csv')
        processor = DataProcessor(df)
        self.df = processor.preprocess()

        self.calculator = RFMSCalculator(self.df)

    def test_calculate_recency(self):
        """
        Tests the calculation of recency.

        Asserts that recency is correctly calculated and returned.
        """
        recency = self.calculator.calculate_recency()
        self.assertIn('Recency', recency.columns)

    def test_calculate_frequency(self):
        """
        Tests the calculation of frequency.

        Asserts that frequency is correctly calculated and returned.
        """
        frequency = self.calculator.calculate_frequency()
        self.assertIn('Frequency', frequency.columns)

    def test_calculate_monetary(self):
        """
        Tests the calculation of monetary.

        Asserts that monetary value is correctly calculated and returned.
        """
        monetary = self.calculator.calculate_monetary()
        self.assertIn('Monetary', monetary.columns)

    def test_calculate_stability(self):
        """
        Tests the calculation of stability.

        Asserts that stability is correctly calculated and returned.
        """
        stability = self.calculator.calculate_stability()
        self.assertIn('Stability', stability.columns)

    def test_calculate_rfms_score(self):
        """
        Tests the calculation of RFMS score.

        Asserts that RFMS score is correctly calculated and returned.
        """
        rfms = self.calculator.calculate_rfms_score()
        self.assertIn('RFMS_Score', rfms.columns)

    def test_define_features_and_target(self):
        """
        Tests the definition of features and target variable.

        Asserts that features and target are correctly defined and returned.
        """
        rfms = self.calculator.calculate_rfms_score()
        features, target = self.calculator.define_features_and_target(rfms)
        self.assertIn('RFMS_Score', features.columns)
        self.assertEqual(target.name, 'Label')


if __name__ == '__main__':
    unittest.main()
