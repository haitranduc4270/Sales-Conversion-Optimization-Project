from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluate_step import evaluate_model


@pipeline(enable_cache=True)
def training_pipeline(data_path: str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    accuracy, precision,recall,f1score,roc_auc = evaluate_model(model, X_test, y_test)
        