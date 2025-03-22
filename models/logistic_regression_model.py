from models.base_ml_model import BaseMLModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from model_registry import register_model, ModelType

@register_model(ModelType.CLASSIFIER)
class LogisticRegressionModel(BaseMLModel):
    """
    Logistic Regression Model that implements the BaseMLModel interface.
    
    Uses scikit-learn's LogisticRegression for training and prediction.
    """

    def __init__(self,**kwargs):
        """
        Logistic Regression Model that implements the BaseMLModel interface.
        
        Uses scikit-learn's LogisticRegression for training and prediction.
        """
        self.model = LogisticRegression(**kwargs)
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data.

        Parameters:
            X (array-like): Training feature matrix
            y (array-like): Training target vector 

        Returns:
            self: Returns the instance itself after training.
        """
        self.X_train = X
        self.y_train = y
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict the target value for the given input data.

        Parameters:
            X (array-like): Feature matrix for which predictions are to be made. 

        Returns:
            array-like: Predicted target values. 
        """
        return self.model.predict(X)

    def accuracy(self, X, y):
        """
        Compute the accuracy of the model on the provided data.

        Parameters:
            X (array-like): Feature matrix for testing.
            y (array-like): True target values for testing.

        Returns:
            float: The accuracy score.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def plot(self):
        """
        Plot may use roc_curve or histogram 
        TBC
        """
