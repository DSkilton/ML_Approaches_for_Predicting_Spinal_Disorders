from enum import Enum
from constants import CLASSIFIER, REGRESSOR

class ModelType(Enum):
    """
    Enumeration for model types 
    """
    CLASSIFIER = CLASSIFIER
    REGRESSOR = REGRESSOR

MODEL_REGISTRY = {
    ModelType.REGRESSOR: [],
    ModelType.CLASSIFIER: []
}

def register_model(model_type: ModelType):
    """
    Decorator to register an ML model class under a specified model tpye.

    Parameters:
        model_type: Either ModelType.CLASSIFIER or ModelType.REGRESSOR.

    Returns:
        Decorator that registers the class in the global MODEL_REGISTRY.
    """
    def decorator(cls):
        MODEL_REGISTRY[model_type].append(cls)
        return cls
    return decorator
