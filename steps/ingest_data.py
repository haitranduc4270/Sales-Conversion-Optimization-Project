import logging


import pandas as pd
from zenml import step

class IngestData:
    """
    Ingesting the data from the data_path.
    """
    def __init__(self, data_path: str,encoding: str='iso-8859-1'):
        """
        Args:
            data_path: path to the data
        """
        self.data_path = data_path
        self.encoding = encoding

    def get_data(self):
        """
        Ingesting the data from the data_path.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path, encoding=self.encoding)

@step
def ingest_df(data_path: str, encoding: str='iso-8859-1' ) -> pd.DataFrame:
    """
    Ingesting the data from the data_path.
    
    Args:
        data_path: path to the data
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_data = IngestData(data_path, encoding=encoding)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e