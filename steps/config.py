from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model Configs"""

    model_name: str = "RandomForestClassifier"
    model_kwargs: dict = {}