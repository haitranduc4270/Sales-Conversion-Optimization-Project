import logging
from abc import ABC, abstractmethod
from typing import Union
from sklearn.preprocessing import LabelEncoder 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass  
    

class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """

    def handle_data(self, data: pd.DataFrame, use_filtered_data: bool = False) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            # Drop unnecessary columns
            data.drop(['ad_id', 'fb_campaign_id'], axis=1, inplace=True)

            # Define the age groups
            age_groups = ['30-34', '35-39', '40-44', '45-49']

            # One-hot encode 'xyz_campaign_id'
            data = pd.get_dummies(data, columns=['xyz_campaign_id'], prefix='campaign', drop_first=True)

            # Initialize the label encoder for 'age'
            label_encoder = LabelEncoder()
            data['Age_Group'] = label_encoder.fit_transform(data['age'])

            # Gender encoding
            data['Gender_Code'] = data['gender'].map({'F': 0, 'M': 1})

            # Interaction Features
            data['Interaction_Imp_Clicks'] = data['Impressions'] * data['Clicks']

            # Spent per Click
            data['Spent_per_Click'] = data['Spent'] / data['Clicks']

            # Total Conversion Rate
            data['Total_Conversion_Rate'] = data['Total_Conversion'] / data['Clicks']

            # Budget Allocation
            data['Budget_Allocation_Imp'] = data['Spent'] / data['Impressions']

            # Ad Performance Metrics
            data['CTR'] = data['Clicks'] / data['Impressions']
            data['Conversion_per_Impression'] = data['Total_Conversion'] / data['Impressions']

            # Drop unnecessary columns
            data.drop(['age', 'gender'], axis=1, inplace=True)

            # Threshold for correlation
            threshold = 0.95
            correlation_matrix = data.corr()

            # Find and drop highly correlated features
            highly_correlated = (correlation_matrix.abs() >= threshold).sum()
            highly_correlated = highly_correlated[highly_correlated > 1].index

            data_filtered = data.drop(columns=highly_correlated)

            # Fill NaN values with 0
            data.fillna(0, inplace=True)

            # Check for infinite values in the DataFrame
            is_inf = np.isinf(data)

            # Replace infinite values with NaN
            data.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Check for infinite values in the DataFrame
            is_inf = np.isinf(data)
            original_data = data.copy()
            imputer = SimpleImputer(strategy='mean')
            data = imputer.fit_transform(data)
            data = pd.DataFrame(data )
            data.columns = original_data.columns
            print(use_filtered_data)
            return data_filtered if use_filtered_data else data

        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e

            

 
class DataDivideStrategy(DataStrategy):
    """
    Strategy for diving datainto train and test
    """

    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data
        """
        try:
            # Prepare the data
            X = data.drop(['Approved_Conversion'], axis=1) 
            y = data['Approved_Conversion']
            # Perform train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e             


class DataCleaning:
    """
    Class for cleaning the data which processes the data and divides it into train and test.
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e 
        
        