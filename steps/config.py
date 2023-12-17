from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model Configs"""

    model_name: str = "GradientBoostingRegressor"
    model_kwargs: dict = {}
