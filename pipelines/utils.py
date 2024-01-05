import logging

import pandas as pd
from src.data_cleaning import DataCleaning, DataPreProcessStrategy


def get_data_for_test():
    try:
        df = pd.read_csv("/home/dhruba/team_project/Sales-Conversion-Optimization-Project/data/raw/KAG_conversion_data.csv")
        df = df.sample(n=100)
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        df.drop(["Total_Conversion"], axis=1, inplace=True)
        print("df from utils",df)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error("Error occurred while processing data: {}".format(e))
        raise e