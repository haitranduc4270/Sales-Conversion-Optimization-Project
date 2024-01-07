import logging 
from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

class Model(ABC):
    """
    Abstract class for all models"""

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training labels
        Returns:
            None
        """
        pass
class GradientBoostingModel(Model):
    """
    Gradient Boosting Regressor model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training labels
            **kwargs: Additional parameters to pass to GradientBoostingRegressor
        Returns:
            None
        """
        try:
            # Instantiate the GradientBoostingRegressor model
            model = GradientBoostingRegressor(**kwargs)

            # Train the model
            model.fit(X_train, y_train)

            # Return the trained model
            return model

        except Exception as e:
            logging.error("Error in training GradientBoostingRegressor model: {}".format(e))
            raise e
        
        
class LinearRegressionModel(Model):
    """
    Gradient Boosting Regressor model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training labels
            **kwargs: Additional parameters to pass to GradientBoostingRegressor
        Returns:
            None
        """
        try:
            # Instantiate the GradientBoostingRegressor model
            model = LinearRegression(**kwargs)

            # Train the model
            model.fit(X_train, y_train)

            # Return the trained model
            return model

        except Exception as e:
            logging.error("Error in training Linear Regression model: {}".format(e))
            raise e
        

class AdaBoostRegressorModel(Model):
    """
    Gradient Boosting Regressor model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training labels
            **kwargs: Additional parameters to pass to GradientBoostingRegressor
        Returns:
            None
        """
        try:
            # Instantiate the GradientBoostingRegressor model
            model = AdaBoostRegressor(**kwargs)

            # Train the model
            model.fit(X_train, y_train)

            # Return the trained model
            return model

        except Exception as e:
            logging.error("Error in training Linear Regression model: {}".format(e))
            raise e
        
class RandomForestRegressorModel(Model):
    """
    Gradient Boosting Regressor model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training labels
            **kwargs: Additional parameters to pass to GradientBoostingRegressor
        Returns:
            None
        """
        try:
            # Instantiate the GradientBoostingRegressor model
            model = RandomForestRegressor(**kwargs)

            # Train the model
            model.fit(X_train, y_train)

            # Return the trained model
            return model

        except Exception as e:
            logging.error("Error in training Linear Regression model: {}".format(e))
            raise e