import logging
from typing import Tuple

import mlflow
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client
from src.evaluation import Accuracy, Conf_matrix,ROC, Precision, Recall, F1_Score


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[
    Annotated[float, "accuracy"],
    # Annotated[int, "confusion_matrix"],
    Annotated[float, "precision"],
    Annotated[float, "recall"],
    Annotated[float, "f1score"],
    Annotated[float, "roc-auc"],
    ]:
    """
    Evaluate the model on the ingested data.
    Args:
        df: the ingested data
    """
    try:
        prediction = model.predict(X_test)
        
        accuracy_class = Accuracy()
        accuracy = accuracy_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("Accuracy", accuracy)


        # conf_matrix_class = Conf_matrix()
        # conf_matrix = conf_matrix_class.calculate_scores(y_test, prediction)
        # mlflow.log_metric("Confusion Matrix", conf_matrix)
        
        precision_class = Precision()
        precision = precision_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("Precision", precision)
        
        
        recall_class = Recall()
        recall = recall_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("Recall", recall)
        
        
        F1_class = F1_Score()
        f1score = F1_class.calculate_scores(y_test,prediction)
        mlflow.log_metric("F1_Score",f1score)
        
        
        # mlflow.log_metric("Recall", recall)
        # mlflow.log_metric("F1-Score", f1)
        # Log other metrics as needed

  
        roc_auc_class = ROC()
        roc_auc= roc_auc_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("ROC-AUC score", roc_auc)


        return accuracy, precision, recall, f1score, roc_auc
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e