import logging
from abc import ABC, abstractmethod
from typing import Union
from sklearn.preprocessing import LabelEncoder 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            # Example preprocessing steps, modify as needed
            value_counts = data['interest'].value_counts()
            total_count = value_counts.sum()
            cum_count = 0
            keep_values = []

            for value, count in value_counts.items():
                cum_count += count
                if cum_count / total_count <= 0.8:
                    keep_values.append(value)
                else:
                    break

            data['pareto_interest'] = data['interest'].apply(lambda x: x if x in keep_values else 'other')

            data['conv1'] = np.where(data['Total_Conversion'] != 0, 1, 0)
            data['conv2'] = np.where(data['Approved_Conversion'] != 0, 1, 0)

            columns_to_one_hot_encode = ['pareto_interest', 'xyz_campaign_id', 'gender', 'age']

            data_dummies = pd.get_dummies(data[columns_to_one_hot_encode], prefix='', prefix_sep='')
            boolean_columns = data_dummies.select_dtypes(include='bool').columns
            data_dummies[boolean_columns] = data_dummies[boolean_columns].astype(int)
            data_dummies = data_dummies[sorted(data_dummies.columns)]
            data = pd.concat([data, data_dummies], axis=1)

            data = data.drop(['age', 'gender', 'xyz_campaign_id', 'fb_campaign_id', 'interest', 'pareto_interest'],
                             axis=1).set_index('ad_id')
            return data
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
            X = data.drop(['Total_Conversion', 'Approved_Conversion', 'conv1', 'conv2'], axis=1) 
            y = data['Total_Conversion']
            print(type(X))
            print(type(y))
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
        
        