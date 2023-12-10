import logging 
from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier


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
    
class RandomForestModel(Model):
    """
    Random Forest model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training labels
            **kwargs: Additional parameters to pass to RandomForestClassifier
        Returns:
            None
        """
        try:
            # Define the hyperparameter grid
            param_grid = {
                'n_estimators': [100, 150, 200],
                'max_depth': [None,10, 20, 30],
                'min_samples_split': [2, 5, 10,15],
                'min_samples_leaf': [1, 2, 4,8]
                # Add other hyperparameters as needed
            }

            # Create a RandomForestClassifier
            rf_model = RandomForestClassifier(random_state=42)

            # Create a GridSearchCV object
            grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

            # Fit the GridSearchCV object to the training data
            grid_search.fit(X_train, y_train)

            # Get the best hyperparameters
            best_params = grid_search.best_params_

            # Use the best hyperparameters to create the final model
            best_rf_model = RandomForestClassifier(**best_params, random_state=42)

            # Train the model on the entire training data
            best_rf_model.fit(X_train, y_train)
                        
            logging.info("Model training completed")
            
            return best_rf_model
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e    
