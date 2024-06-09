#!/usr/bin/python3

"""
This module contains the RFMSCalculator class for calculating RFMS scores.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.data_processor import DataProcessor


class RFMSCalculator(DataProcessor):
    """
    Class for calculating RFMS (
    Recency, Frequency, Monetary, Stability) scores.

    Inherits from BaseDataProcessor.

    Attributes:
        df (pd.DataFrame): The DataFrame to be processed.
    """
    def calculate_recency(self):
        """
        Calculates the recency for each account.

        Returns:
            pd.DataFrame: A DataFrame with recency values.
        """
        self.df['DaysSinceLastTransaction'] = (
            pd.Timestamp.now(tz='UTC') - self.df['TransactionStartTime']
        ).dt.days
        recency = self.df.groupby(
            'AccountId_LabelEnc')['DaysSinceLastTransaction'].min(
            ).reset_index()
        recency.columns = ['AccountId_LabelEnc', 'Recency']
        return recency

    def calculate_frequency(self):
        """
        Calculates the frequency of transactions for each account.

        Returns:
            pd.DataFrame: A DataFrame with frequency values.
        """
        frequency = self.df.groupby(
            'AccountId_LabelEnc')['TransactionId_LabelEnc'].count(
            ).reset_index()
        frequency.columns = ['AccountId_LabelEnc', 'Frequency']
        return frequency

    def calculate_monetary(self):
        """
        Calculates the average monetary value of transactions for each account.

        Returns:
            pd.DataFrame: A DataFrame with monetary values.
        """
        monetary = self.df.groupby(
            'AccountId_LabelEnc')['Amount'].mean().reset_index()
        monetary.columns = ['AccountId_LabelEnc', 'Monetary']
        return monetary

    def calculate_stability(self):
        """
        Calculates the stability (standard deviation) of transaction
        amounts for each account.

        Returns:
            pd.DataFrame: A DataFrame with stability values.
        """
        stability = self.df.groupby(
            'AccountId_LabelEnc')['Amount'].std().reset_index()
        stability.columns = ['AccountId_LabelEnc', 'Stability']
        return stability

    def calculate_rfms_score(self):
        """
        Calculates the RFMS score for each account by combining
        recency, frequency, monetary, and stability metrics.

        Returns:
            pd.DataFrame: A DataFrame with RFMS scores.
        """
        recency = self.calculate_recency()
        frequency = self.calculate_frequency()
        monetary = self.calculate_monetary()
        stability = self.calculate_stability()

        rfms = recency.merge(
            frequency, on='AccountId_LabelEnc').merge(
                monetary, on='AccountId_LabelEnc').merge(
                    stability, on='AccountId_LabelEnc')

        scaler = StandardScaler()
        rfms[['Recency', 'Frequency', 'Monetary', 'Stability']] = \
            scaler.fit_transform(
                rfms[['Recency', 'Frequency', 'Monetary', 'Stability']])

        rfms['RFMS_Score'] = 0.25 * rfms['Recency'] + 0.25 * \
            rfms['Frequency'] + 0.25 * rfms['Monetary'] + 0.25 * \
            rfms['Stability']

        # Set threshold
        threshold = rfms['RFMS_Score'].median()

        # Assign labels
        rfms['Label'] = np.where(
            rfms['RFMS_Score'] >= threshold, 'good', 'bad')

        return rfms

    def calculate_woe_iv(self, df, feature, target):
        """
        Calculate Weight of Evidence (WoE) and Information Value (IV)
        for a given
        feature in a DataFrame.

        Args:
            df (pandas.DataFrame): The input DataFrame containing the data.
            feature (str): The name of the feature/column for which to
        calculate WoE and IV.
        target (str): The name of the target column which contains the
        binary class labels ('good' or 'bad').

        Returns:
            pandas.DataFrame: A DataFrame with the calculated WoE and
        IV values for each category of the given feature.

        Note:
            - The target column should contain only two unique values:
        'good' and 'bad'.
            - A small number (eps = 0.00001) is added to prevent division
        by zero.
        - In finite WoE values are replaced with 0.
        - The returned DataFrame includes the WoE for each category and
        the overall IV value.
        """
        # Function to calculate WoE and IV
        eps = 0.00001  # a small number to prevent division by zero
        df = df.copy()
        df['good'] = np.where(df[target] == 'good', 1, 0)
        df['bad'] = np.where(df[target] == 'bad', 1, 0)
        grouped = df.groupby(feature).agg({'good': 'sum', 'bad': 'sum'})
        grouped['total'] = grouped['good'] + grouped['bad']
        grouped['percent_good'] = grouped['good'] / grouped['good'].sum()
        grouped['percent_bad'] = grouped['bad'] / grouped['bad'].sum()
        grouped['WoE'] = np.log(
            (grouped['percent_good'] + eps) / (grouped['percent_bad'] + eps))
        grouped['IV'] = (
            grouped['percent_good'] - grouped['percent_bad']) * grouped['WoE']
        grouped = grouped.replace([np.inf, -np.inf], 0)
        grouped['IV'] = grouped['IV'].sum()

        return grouped[['WoE', 'IV']]

    def define_features_and_target(self, rfms):
        """
        Defines features and target variable for the model.

        Args:
            rfms (pd.DataFrame): The DataFrame with RFMS scores.

        Returns:
            tuple: A tuple containing features (pd.DataFrame) and
        target (pd.Series).
        """
        features = rfms.drop(columns=['AccountId_LabelEnc', 'Label'])
        target = rfms['Label']

        # self.handle_missing_values(['Stability', 'RFMS_Score'], rfms)
        features['Stability'].fillna(
            features['Stability'].mean(), inplace=True)
        features['RFMS_Score'].fillna(
            features['RFMS_Score'].mean(), inplace=True)

        return features, target
