from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin

class TextProcessorInterface(BaseEstimator, TransformerMixin, ABC):
    """
    Interface for text processing that converts raw text into numeric features.

    Implementations of this interface should provide a mechanism to fit a corpus
    of text and transform new text data into the corresponding numeric representation.
    """

    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit the text processor on the provided text data.

        Parameters:
            X (iterable): A  collection of text documents.
            y: Ignored; present for compatability with scikit-learn.

        Returns:
            self
        """
        pass

    @abstractmethod
    def transform(self, X):
        """
        Transform the input text data into number features. 
        
        Parameters:
            X (iterable): A collection of text documents.
            
        Returns: 
            Transformed feature matrix.
        """
        pass