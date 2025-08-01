import os
import numpy as np
import logging
from typing import Any

from utils import load_model, load_california_housing_data, calculate_metrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """Model predictor class for running inference."""
    
    def __init__(self, model_path: str = "model.joblib", scaler_path: str = "scaler.joblib"):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the feature scaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        
    def load_model_and_scaler(self):
        """Load the trained model and scaler."""
        try:
            # Load model
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from {self.model_path}")
            self.model = load_model(self.model_path)
            
            # Load scaler if it exists
            if os.path.exists(self.scaler_path):
                logger.info(f"Loading scaler from {self.scaler_path}")
                self.scaler = load_model(self.scaler_path)
            else:
                logger.warning(f"Scaler file not found: {self.scaler_path}. Using raw features.")
                self.scaler = None
                
        except Exception as e:
            logger.error(f"Failed to load model or scaler: {str(e)}")
            raise
    
    def preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess features using the loaded scaler.
        
        Args:
            X: Input features
            
        Returns:
            Preprocessed features
        """
        if self.scaler is not None:
            return self.scaler.transform(X)
        else:
            return X
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_scaler() first.")
        
        # Preprocess features
        X_processed = self.preprocess_features(X)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        return predictions
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        r2, mse = calculate_metrics(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        
        metrics = {
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'num_samples': len(y_test)
        }
        
        return metrics


def run_model_verification():
    """
    Run complete model verification including loading, prediction, and evaluation.
    """
    logger.info("Starting model verification...")
    
    try:
        # Initialize predictor
        predictor = ModelPredictor()
        
        # Load model and scaler
        predictor.load_model_and_scaler()
        
        # Load test data
        logger.info("Loading California housing test data...")
        _, X_test, _, y_test = load_california_housing_data()
        
        # Make predictions on full test set
        logger.info("Making predictions on test set...")
        y_pred = predictor.predict(X_test)
        
        # Evaluate model performance
        logger.info("Evaluating model performance...")
        metrics = predictor.evaluate_model(X_test, y_test)
        
        # Print results
        print("="*60)
        print("MODEL VERIFICATION RESULTS")
        print("="*60)
        print(f"Model Type: {type(predictor.model).__name__}")
        print(f"Test Samples: {metrics['num_samples']:,}")
        print(f"R² Score: {metrics['r2_score']:.4f}")
        print(f"Mean Squared Error: {metrics['mse']:.4f}")
        print(f"Root Mean Squared Error: {metrics['rmse']:.4f}")
        print(f"Mean Absolute Error: {metrics['mae']:.4f}")
        print("="*60)
        
        # Show sample predictions
        print("\\nSAMPLE PREDICTIONS:")
        print("-" * 40)
        sample_size = min(10, len(y_test))
        for i in range(sample_size):
            print(f"Sample {i+1:2d}: Predicted={y_pred[i]:.3f}, Actual={y_test[i]:.3f}, "
                  f"Error={abs(y_pred[i] - y_test[i]):.3f}")
        
        if len(y_test) > sample_size:
            print(f"... and {len(y_test) - sample_size:,} more samples")
        
        print("-" * 40)
        
        # Model parameter info
        if hasattr(predictor.model, 'coef_') and hasattr(predictor.model, 'intercept_'):
            print(f"\\nMODEL PARAMETERS:")
            print(f"Number of features: {len(predictor.model.coef_)}")
            print(f"Intercept: {predictor.model.intercept_:.4f}")
            print(f"Coefficients shape: {predictor.model.coef_.shape}")
            print(f"Coefficient range: [{np.min(predictor.model.coef_):.4f}, {np.max(predictor.model.coef_):.4f}]")
        
        # Success message
        print("\\n✅ Model verification completed successfully!")
        print(f"📊 R² Score: {metrics['r2_score']:.4f}")
        print(f"📈 RMSE: {metrics['rmse']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model verification failed: {str(e)}")
        print(f"\\n❌ Model verification failed: {str(e)}")
        return False


def main():
    """Main function for running prediction script."""
    print("Starting Docker Model Verification...")
    print("=" * 60)
    
    success = run_model_verification()
    
    if success:
        print("\\n🎉 Docker container verification successful!")
        exit(0)
    else:
        print("\\n💥 Docker container verification failed!")
        exit(1)


if __name__ == "__main__":
    main()
