from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator

class PersistenceInterface(ABC):
    """
    Abstract interface for persistence operations on machine learning models.

    This interface defines the contract for saving and loading a scikit-learn BaseEstimator model.
    Any concrete implementation should provide methods to persist (save) and retrieve (load) models.
    """
    @classmethod
    @abstractmethod
    def save(cls, model: BaseEstimator, filepath: str):
        """
        Persist the given machine learning model to the specified file path.

        Parameters:
            model (BaseEstimator): The machine learning model to be saved.
            filepath (str): The file path where the model should be stored.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> BaseEstimator:
        """
        Load and return a machine learning model from the specified file path.

        Parameters:
            filepath (str): The file path from which to load the model.

        Returns:
            BaseEstimator: The machine learning model loaded from the file.
        """
        pass