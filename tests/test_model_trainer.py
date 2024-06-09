#!/usr/bin/python3

"""
This module contains unit tests for the ModelTrainer class.
"""

import unittest
import pandas as pd
from sklearn.datasets import make_classification
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.rfms_calculator import RFMSCalculator


class TestModelTrainer(unittest.TestCase):
    """
    Unit tests for the ModelTrainer class.
    """

    def setUp(self):
        """
        Initializes test fixtures before each test method.

        Creates a sample dataset and initializes a ModelTrainer instance.
        """
        df = pd.read_csv('data/data.csv')

        # Process data
        processor = DataProcessor(df)
        df = processor.preprocess()

        # Calculate RFMS scores
        rfms_calculator = RFMSCalculator(df)
        rfms = rfms_calculator.calculate_rfms_score()

        # Define features and target
        features, target = rfms_calculator.define_features_and_target(rfms)
        self.trainer = ModelTrainer(features, target)

    def test_split_data(self):
        """
        Tests the splitting of data into training and testing sets.

        Asserts that the data is correctly split.
        """
        self.trainer.split_data()
        self.assertEqual(len(self.trainer.X_train), 2543)
        self.assertEqual(len(self.trainer.X_test), 1090)

    def test_train_models(self):
        """
        Tests the training of machine learning models.

        Asserts that the models are correctly trained.
        """
        self.trainer.split_data()
        self.trainer.train_models()
        self.assertIsNotNone(self.trainer.logistic_model)
        self.assertIsNotNone(self.trainer.random_forest_model)
        self.assertIsNotNone(self.trainer.gbm_model)

    def test_hyperparameter_tuning(self):
        """
        Tests hyperparameter tuning using GridSearchCV.

        Asserts that the hyperparameter tuning is correctly performed.
        """
        self.trainer.split_data()
        self.trainer.train_models()
        self.trainer.hyperparameter_tuning()
        self.assertIsNotNone(self.trainer.logistic_grid.best_estimator_)
        self.assertIsNotNone(self.trainer.rf_grid.best_estimator_)
        self.assertIsNotNone(self.trainer.gbm_grid.best_estimator_)

    def test_evaluate_model(self):
        """
        Tests the evaluation of a trained model.

        Asserts that the model is correctly evaluated and metrics are returned.
        """
        self.trainer.split_data()
        self.trainer.train_models()
        self.trainer.hyperparameter_tuning()
        logistic_metrics = self.trainer.evaluate_model(
            self.trainer.logistic_grid.best_estimator_)
        self.assertEqual(len(logistic_metrics), 5)
        self.assertTrue(all(
            isinstance(metric, float) for metric in logistic_metrics))


if __name__ == '__main__':
    unittest.main()
