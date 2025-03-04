from abc import ABC, abstractmethod

class BaseMLModel(ABC):
    """
    Abstract base class for machine learning models.

    This class defines a standard interface for training, predicting, evaluating, and 
    visualizing the performance of various machine learning models. All subclasses must 
    implement the abstract methods defined below.
    """

    def __init__(self):
        """
        Initialise the BaseMLModel.

        This constructor can be extended by subclasses to initialise any model-specific parameters.
        """
        pass

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters:
            X (array-like): The feature matrix for training.
            y (array-like): The target vector for training.

        Returns:
            self: Returns the instance itself.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict the target values for given input data.

        Parameters:
            X (array-like): The feature matrix for which to make predictions.

        Returns:
            array-like: Predicted target values.
        """
        pass

    @abstractmethod
    def accuracy(self):
        """
        Compute the accuracy of the model.

        This method should calculate and return a metric representing the model's accuracy, 
        based on predictions and actual target values stored internally or passed via an evaluation process.

        Returns:
            float: The accuracy score of the model.
        """
        pass

    @abstractmethod
    def plot(self):
        """
        Plot relevant aspects of the model's performance.

        This method should generate and display plots that help visualize the performance 
        or characteristics of the model (e.g., decision boundaries, learning curves, or feature importances).
        """
        pass
