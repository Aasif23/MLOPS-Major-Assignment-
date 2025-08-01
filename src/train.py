import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import logging

from utils import (
    load_california_housing_data,
    calculate_metrics,
    save_model,
    print_model_info
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_linear_regression_model():
    """
    Train a Linear Regression model on the California housing dataset.
    
    Returns:
        tuple: (trained_model, X_test, y_test, r2_score, mse)
    """
    logger.info("Starting model training...")
    
    # Load and split data
    X_train, X_test, y_train, y_test = load_california_housing_data()
    
    # Feature scaling for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the Linear Regression model
    logger.info("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2, train_mse = calculate_metrics(y_train, y_pred_train)
    test_r2, test_mse = calculate_metrics(y_test, y_pred_test)
    
    # Print results
    print("="*50)
    print("MODEL TRAINING RESULTS")
    print("="*50)
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Training MSE (Loss): {train_mse:.4f}")
    print(f"Training RMSE: {np.sqrt(train_mse):.4f}")
    print("-"*50)
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Test MSE (Loss): {test_mse:.4f}")
    print(f"Test RMSE: {np.sqrt(test_mse):.4f}")
    print("="*50)
    
    # Print model coefficients info
    print(f"Number of features: {len(model.coef_)}")
    print(f"Model intercept: {model.intercept_:.4f}")
    print(f"Model coefficients shape: {model.coef_.shape}")
    
    # Save the trained model
    model_filepath = "model.joblib"
    save_model(model, model_filepath)
    
    # Save the scaler as well for consistent preprocessing
    scaler_filepath = "scaler.joblib"
    save_model(scaler, scaler_filepath)
    
    logger.info("Model training completed successfully!")
    
    return model, scaler, X_test_scaled, y_test, test_r2, test_mse


def validate_training_requirements():
    """
    Validate that training meets the assignment requirements.
    """
    logger.info("Validating training requirements...")
    
    # Check if model file exists
    if not os.path.exists("model.joblib"):
        raise FileNotFoundError("Model file not found after training!")
    
    # Load and validate model
    model = joblib.load("model.joblib")
    
    # Check if it's a LinearRegression model
    if not isinstance(model, LinearRegression):
        raise TypeError("Model is not a LinearRegression instance!")
    
    # Check if model has required attributes
    if not hasattr(model, 'coef_') or not hasattr(model, 'intercept_'):
        raise AttributeError("Model missing required attributes (coef_ or intercept_)!")
    
    logger.info("All training requirements validated successfully!")


if __name__ == "__main__":
    try:
        # Train the model
        model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
        
        # Validate requirements
        validate_training_requirements()
        
        # Success message
        print("\\n✅ Training completed successfully!")
        print(f"📊 Final Test R² Score: {r2:.4f}")
        print(f"📉 Final Test MSE: {mse:.4f}")
        print(f"💾 Model saved as: model.joblib")
        print(f"🔧 Scaler saved as: scaler.joblib")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
