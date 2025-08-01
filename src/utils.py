import numpy as np
import joblib
from typing import Tuple, Union, Any
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_california_housing_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split California housing dataset.
    
    Args:
        test_size: Proportion of the dataset to include in the test split
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Loading California housing dataset...")
    
    # Load dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    logger.info(f"Dataset shape: {X.shape}, Target shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Calculate R² score and MSE for predictions.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Tuple of (r2_score, mse)
    """
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    return r2, mse


def save_model(model: Any, filepath: str) -> None:
    """
    Save model using joblib.
    
    Args:
        model: Trained model to save
        filepath: Path where to save the model
    """
    logger.info(f"Saving model to {filepath}")
    joblib.dump(model, filepath)


def load_model(filepath: str) -> Any:
    """
    Load model using joblib.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {filepath}")
    return joblib.load(filepath)


def validate_model_parameters(model: Any) -> bool:
    """
    Validate that model has required parameters for Linear Regression.
    
    Args:
        model: Model to validate
        
    Returns:
        True if model has required parameters, False otherwise
    """
    try:
        # Check if model has coefficients and intercept
        has_coef = hasattr(model, 'coef_')
        has_intercept = hasattr(model, 'intercept_')
        return has_coef and has_intercept
    except Exception:
        return False


def print_model_info(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Print model information and performance metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2, mse = calculate_metrics(y_test, y_pred)
    
    print(f"Model Type: {type(model).__name__}")
    print(f"Number of features: {len(model.coef_)}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")
