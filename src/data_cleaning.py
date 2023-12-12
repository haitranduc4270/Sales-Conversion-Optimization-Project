import logging
from abc import ABC, abstractmethod
from typing import Union
from sklearn.preprocessing import LabelEncoder 
from imblearn.under_sampling import RandomUnderSampler

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
            for col in data.select_dtypes(include=[np.number]).columns:
                cols_with_outliers = []
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                total_outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                if total_outliers > 0:
                    cols_with_outliers.append(col)
                print(f'{col} has total outliers: ', total_outliers)
            
            # Replace outliers
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                
            # Apply one-hot encoding for gender
            #data = pd.get_dummies(data, columns=['gender'], prefix='gender', drop_first=True)
            #data['gender_M'] = data['gender_M'].astype(int)
             
                 
            # Apply one-hot encoding for gender
            data = pd.get_dummies(data, columns=['gender'], prefix='gender', drop_first=True)
            data['gender_M'] = data['gender_M'].astype(int) 
                 
                
                # Label encode age
            le = LabelEncoder()
            data['age_encoded'] = le.fit_transform(data['age']) 
            
            # for col in data:
            #     cols_with_outliers = []
            #     q1 = data[col].quantile(0.25)
            #     q3 = data[col].quantile(0.75)
            #     iqr = q3 - q1
            #     lower_bound = q1 - 1.5 * iqr
            #     upper_bound = q3 + 1.5 * iqr

            #     total_outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            #     if total_outliers > 0:
            #         cols_with_outliers.append(col)
            #     print(f'{col} has total outliers: ', total_outliers)
            
            #     # Replace outliers
            #     data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Feature engineering
            data['Approved_Conversion'] = data['Approved_Conversion'].apply(lambda x: 1 if x > 0 else 0)
            data['spent_clicks_campaign_interaction'] = data['Spent'] * data['Clicks'] * data['xyz_campaign_id']
            data['spent_clicks_age_interaction'] = data['Spent'] * data['Clicks'] * data['age_encoded']
            data['spent_clicks_interest_interaction'] = data['Spent'] * data['Clicks'] * data['interest']
            data['spent_per_click'] = data['Spent'] / (data['Clicks'] + 1)
            # Drop unnecessary columns
            columns_to_drop = [ 'age' , 'xyz_campaign_id', 'ad_id', 'fb_campaign_id','Impressions', 'Spent', 'Clicks']
            data = data.drop(columns=columns_to_drop,axis=1)
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
            X = data.drop("Approved_Conversion", axis=1)
            y = data["Approved_Conversion"]
            # Apply undersampling
            under_sampler = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = under_sampler.fit_resample(X, y)
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
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
        
        