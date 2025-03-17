from sklearn.base import BaseEstimator
from persistence_interface import PersistenceInterface
import joblib

class Persistence(PersistenceInterface):
    """
    Concrete implementation of PersistenceInterface using joblib for model persistence.

    This class provides methods to save and load a scikit-learn BaseEstimator model
    to and from a file, allowing for easy persistence and retrieval of trained models.
    """

    def save(self, model: BaseEstimator, filepath: str):
        """
        Save a scikit-learn model to the specified file path using joblib.

        Parameters:
            model (BaseEstimator): The scikit-learn model to be persisted.
            filepath (str): The file path where the model should be saved.
        """
        joblib.dump(model, filepath)

    def load(self, filepath:str) -> BaseEstimator:
        """
        Load and return a scikit-learn model from the specified file path using joblib.

        Parameters:
            filepath (str): The file path from which the model should be loaded.

        Returns:
            BaseEstimator: The scikit-learn model loaded from the file.
        """
        return joblib.load(filepath)

