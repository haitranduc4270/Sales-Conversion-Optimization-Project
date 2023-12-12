import logging
from abc import ABC, abstractmethod

import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score,precision_score,recall_score,f1_score


class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation our model
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates scores for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            None
        """
        pass
    
    
class Accuracy(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Accuracy")
            accuracy = accuracy_score(y_true, y_pred)
            logging.info("Accuracy: {}".format(accuracy))
            return accuracy
        except Exception as e:
            logging.error("Error in calculating Accuracy: {}".format(e))
            raise e
        

class Conf_matrix(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Confusion Matrix")
            conf_matrix = confusion_matrix(y_true, y_pred)
            logging.info("Confusion Matrix: {}".format(conf_matrix))
            return conf_matrix
        except Exception as e:
            logging.error("Error in calculating Confusion Matrix: {}".format(e))
            raise e
        
        
class Precision(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Precision Score")
            precision = precision_score(y_true, y_pred)
            logging.info("Precision Score: {}".format(precision))
            return precision
        except Exception as e:
            logging.error("Error in calculating Precision: {}".format(e))
            raise 
        
class Recall(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Recall")
            recall = recall_score(y_true, y_pred)
            logging.info("Recall: {}".format(recall))
            return recall
        except Exception as e:
            logging.error("Error in calculating Recall: {}".format(e))
            raise 


class F1_Score(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating F1 Score")
            f1score= f1_score(y_true, y_pred)
            logging.info("F1_Score: {}".format(f1score))
            return f1score
        except Exception as e:
            logging.error("Error in calculating F1_Score: {}".format(e))
            raise 
                
  
class ROC(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating ROC-AUC")
            roc = roc_auc_score(y_true, y_pred)
            logging.info("ROC-AUC: {}".format(roc))
            return roc
        except Exception as e:
            logging.error("Error in calculating ROC-AUC: {}".format(e))
            raise e