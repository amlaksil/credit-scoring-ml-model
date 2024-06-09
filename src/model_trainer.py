#!/usr/bin/python3

"""
This module contains the ModelTrainer class for training and evaluating
machine learning models.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score


class ModelTrainer:
    """
    Class for training and evaluating machine learning models.

    Attributes:
        X (pd.DataFrame): Features for training the model.
        y (pd.Series): Target variable for training the model.
    """

    def __init__(self, X, y):
        """
        Initializes the ModelTrainer with features and target variable.

        Args:
            X (pd.DataFrame): Features for training the model.
            y (pd.Series): Target variable for training the model.
        """
        self.X = X
        self.y = y

    def split_data(self):
        """
        Splits the data into training and testing sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(
                self.X, self.y, test_size=0.3, random_state=42)

    def train_models(self):
        """
        Trains logistic regression, random forest, and gradient
        boosting models on the training data.
        """
        self.logistic_model = LogisticRegression(random_state=42)
        self.random_forest_model = RandomForestClassifier(random_state=42)
        self.gbm_model = GradientBoostingClassifier(random_state=42)

        self.logistic_model.fit(self.X_train, self.y_train)
        self.random_forest_model.fit(self.X_train, self.y_train)
        self.gbm_model.fit(self.X_train, self.y_train)

    def hyperparameter_tuning(self):
        """
        Performs hyperparameter tuning using GridSearchCV for logistic
        regression, random forest, and gradient boosting models.
        """
        logistic_params = {'C': [0.1, 1, 10, 100]}
        self.logistic_grid = GridSearchCV(
            self.logistic_model, logistic_params, cv=5, scoring='roc_auc')
        self.logistic_grid.fit(self.X_train, self.y_train)

        rf_params = {'n_estimators': [50, 100, 200],
                     'max_depth': [None, 10, 20, 30]}
        self.rf_grid = GridSearchCV(
            self.random_forest_model, rf_params, cv=5, scoring='roc_auc')
        self.rf_grid.fit(self.X_train, self.y_train)

        gbm_params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 4, 5]}
        self.gbm_grid = GridSearchCV(
            self.gbm_model, gbm_params, cv=5, scoring='roc_auc')
        self.gbm_grid.fit(self.X_train, self.y_train)

    def evaluate_model(self, model, threshold=0.5):
        """
        Evaluates a trained model on the testing data.

        Args:
            model: The trained model to evaluate.
            threshold (float): The threshold for converting probabilities
        to class labels.

        Returns:
            tuple: A tuple containing accuracy, precision, recall
        F1 score, and ROC-AUC score.
        """
        y_prob = model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        y_pred_mapped = ['good' if p == 1 else 'bad' for p in y_pred]

        accuracy = accuracy_score(self.y_test, y_pred_mapped)
        precision = precision_score(
            self.y_test, y_pred_mapped, pos_label='good')
        recall = recall_score(self.y_test, y_pred_mapped, pos_label='good')
        f1 = f1_score(self.y_test, y_pred_mapped, pos_label='good')
        roc_auc = roc_auc_score(self.y_test, y_prob)

        return accuracy, precision, recall, f1, roc_auc
