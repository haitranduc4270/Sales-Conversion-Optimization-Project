from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model Configs"""

    model_name: str = "FCNN"
    model_kwargs: dict = {}
