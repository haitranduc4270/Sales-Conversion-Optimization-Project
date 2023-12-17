import logging 
from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
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
    
# class FCNNModel(Model):
#     """
#     Random Forest model
#     """
#     def train(self, X_train, y_train, **kwargs):
#         """
#         Trains the model
#         Args:
#             X_train: Training data
#             y_train: Training labels
#             **kwargs: Additional parameters to pass to RandomForestClassifier
#         Returns:
#             None
#         """
#         try:
#             # Define the CNN model architecture
#             class FCNN(nn.Module):
#                 def __init__(self, input_size, hidden_size,output_size, **kwargs):
#                     super(FCNN, self).__init__()
#                     # Define your CNN layers here

#                     # Example architecture
#                     self.fc1 = nn.Linear(input_size, hidden_size)
#                     self.relu = nn.ReLU()
#                     self.fc2 = nn.Linear(hidden_size, output_size)

#                 def forward(self, x):
#                     out = self.fc1(x)
#                     out = self.relu(out)
#                     out = self.fc2(out)
#                     return out

#             # Define the hyperparameters
#             input_size = X_train.shape[1]
#             hidden_size = 73
#             output_size = 1
#             learning_rate = 0.001
#             num_epochs = 900

#             # Instantiate the CNN model
#             model = FCNN(input_size, hidden_size, output_size)

#             # Define the loss function and optimizer
#             criterion = nn.MSELoss()
#             optimizer = optim.Adam(model.parameters(), lr=learning_rate)


#             X_train_np = X_train.to_numpy()  # Convert X_train to NumPy array
#             y_train_np = y_train.to_numpy()
            
#             # Convert X_train and y_train to PyTorch tensors if not already
#             X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
#             y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)  # Assuming y_train contains class indices

#             # Training loop

#             for epoch in range(num_epochs):
#                 # Forward pass
#                 outputs = model(X_train_tensor)
#                 loss = criterion(outputs, y_train_tensor)

#                 # Backward and optimize
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#             # Return the trained CNN model
#             return model

        except Exception as e:
            logging.error("Error in training FCNN model: {}".format(e))
            raise e