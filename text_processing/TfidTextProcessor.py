from sklearn.feature_extraction.text import TfidfVectorizer
from text_processing.text_processor import TextProcessorInterface

class TfidTextProcessor(TextProcessorInterface):
    """
    TF-IDF implementation of the TextProcessorInterface.

    This class wraps scikit-learns TfidfVectorizer to convert text documents into
    TD-IFD weight numberic features. 
    """

    def __init__(self, **kwargs):
        """
        Initialise the TD-IDF text processor.

        Parameters:
            **kwargs: Additional keyword args from TfidfVectorizer.
        """
        self.vectorizer = TfidfVectorizer(**kwargs)

    def fit(self, X, y=None):
        """
        Fit the TF-IDF vectorizer on the provided text data.

        Parameters:
            X (iterable): A collection of text documents.
            y: Ignored, present for scikit-learn compat

        Returns: 
            self
        """
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        """
        Transform the text data into TD-IDF weight numeric feature matrix.

        Parameters:
            X (iterable): A collection of text documents.

        Returns: The TF-IDF matrix (sparse matrix by default).
        """
        return self.vectorizer.transform(X)
