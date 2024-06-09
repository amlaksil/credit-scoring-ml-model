#!/usr/bin/python3
"""
This module contains the DataProcessor class for processing raw data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


class DataProcessor:
    """
    Class for processing raw data into a format suitable for machine learning.

    Inherits from BaseDataProcessor.

    Attributes:
        df (pd.DataFrame): The DataFrame to be processed.
    """

    def __init__(self, df):
        """
        Initializes the DataProcessor with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be processed.
        """
        self.df = df

    def calculate_aggregate_features(self):
        """
        Calculates aggregate features for the DataFrame and merges
        them back into the DataFrame.
        """
        aggregate_features = self.df.groupby('AccountId').agg({
            'Amount': ['sum', 'mean', 'std', 'count'],
        }).reset_index()

        aggregate_features.columns = [
            'AccountId', 'TotalTransactionAmount', 'AverageTransactionAmount',
            'StdDevTransactionAmount', 'TransactionCount'
        ]

        self.df = self.df.merge(aggregate_features, on='AccountId', how='left')

    def extract_time_features(self):
        """
        Extracts time-based features from the 'TransactionStartTime' column.
        """
        self.df['TransactionStartTime'] = pd.to_datetime(
            self.df['TransactionStartTime'])
        self.df['TransactionHour'] = self.df['TransactionStartTime'].dt.hour
        self.df['TransactionDay'] = self.df['TransactionStartTime'].dt.day
        self.df['TransactionMonth'] = self.df['TransactionStartTime'].dt.month
        self.df['TransactionYear'] = self.df['TransactionStartTime'].dt.year

    def encode_categorical_features(self):
        """
        Encodes categorical features using Label Encoding and One-Hot Encoding.
        """
        label_encoders = {}
        for col in [
                'TransactionId', 'BatchId', 'AccountId',
                'SubscriptionId', 'CustomerId', 'CurrencyCode']:
            le = LabelEncoder()
            self.df[col + '_LabelEnc'] = le.fit_transform(self.df[col])
            label_encoders[col] = le
            self.df.drop(col, axis=1, inplace=True)

        self.df = pd.get_dummies(
            self.df, columns=['ProductCategory', 'ChannelId',
                              'ProviderId', 'CountryCode', 'ProductId'])

    def normalize_features(self):
        """
        Normalizes numerical features to have a mean of 0 and a
        standard deviation of 1.
        """
        numerical_features = ['TotalTransactionAmount',
                              'AverageTransactionAmount',
                              'StdDevTransactionAmount', 'TransactionCount',
                              'TransactionHour', 'TransactionDay',
                              'TransactionMonth', 'TransactionYear']
        scaler = StandardScaler()
        self.df[numerical_features] = scaler.fit_transform(
            self.df[numerical_features])

    def preprocess(self):
        """
        Executes the full preprocessing pipeline on the DataFrame.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        self.calculate_aggregate_features()
        self.extract_time_features()
        self.encode_categorical_features()

        # Impute missing values for numerical features
        self.df['StdDevTransactionAmount'].fillna(
                self.df['StdDevTransactionAmount'].mean(), inplace=True)

        self.normalize_features()
        return self.df
