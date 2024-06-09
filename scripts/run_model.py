#!/usr/bin/python3

"""
This script runs the full data processing, feature engineering, and
model training pipeline.
"""

import pandas as pd
from src.data_processor import DataProcessor
from src.rfms_calculator import RFMSCalculator
from src.model_trainer import ModelTrainer


def main():
    """
    Main function to load data, process data, calculate RFMS scores
    define features and target, train models, and evaluate models.
    """
    # Load data
    df = pd.read_csv('data/data.csv')

    # Process data
    processor = DataProcessor(df)
    df = processor.preprocess()

    # Calculate RFMS scores
    rfms_calculator = RFMSCalculator(df)
    rfms = rfms_calculator.calculate_rfms_score()

    # Calculate WoE
    woe_iv = rfms_calculator.calculate_woe_iv(rfms, 'RFMS_Score', 'Label')
    print(woe_iv)

    # Define features and target
    features, target = rfms_calculator.define_features_and_target(rfms)

    # Train and evaluate models
    trainer = ModelTrainer(features, target)
    trainer.split_data()
    trainer.train_models()
    trainer.hyperparameter_tuning()

    # Evaluate models
    logistic_metrics = trainer.evaluate_model(
        trainer.logistic_grid.best_estimator_)
    rf_metrics = trainer.evaluate_model(trainer.rf_grid.best_estimator_)
    gbm_metrics = trainer.evaluate_model(trainer.gbm_grid.best_estimator_)

    print(f"Logistic Regression: {logistic_metrics}")
    print(f"Random Forest: {rf_metrics}")
    print(f"Gradient Boosting Machines: {gbm_metrics}")


if __name__ == "__main__":
    main()
