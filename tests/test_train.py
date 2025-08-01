import os
import sys
import pytest
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Add src directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    load_california_housing_data,
    calculate_metrics,
    save_model,
    load_model,
    validate_model_parameters
)
from train import train_linear_regression_model, validate_training_requirements


class TestDatasetLoading:
    """Test suite for dataset loading functionality."""
    
    def test_load_california_housing_data_returns_correct_shapes(self):
        """Test that dataset loading returns correct shapes."""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        # Check that we get 4 arrays
        assert len([X_train, X_test, y_train, y_test]) == 4
        
        # Check shapes are consistent
        assert X_train.shape[0] == y_train.shape[0]  # Same number of training samples
        assert X_test.shape[0] == y_test.shape[0]    # Same number of test samples
        assert X_train.shape[1] == X_test.shape[1]   # Same number of features
        
        # Check feature dimension (California housing has 8 features)
        assert X_train.shape[1] == 8
        assert X_test.shape[1] == 8
    
    def test_load_california_housing_data_with_custom_test_size(self):
        """Test dataset loading with custom test size."""
        test_size = 0.3
        X_train, X_test, y_train, y_test = load_california_housing_data(test_size=test_size)
        
        total_samples = X_train.shape[0] + X_test.shape[0]
        actual_test_ratio = X_test.shape[0] / total_samples
        
        # Check test size is approximately correct (within 1% tolerance)
        assert abs(actual_test_ratio - test_size) < 0.01
    
    def test_load_california_housing_data_reproducibility(self):
        """Test that dataset loading is reproducible with same random state."""  
        X_train1, X_test1, y_train1, y_test1 = load_california_housing_data(random_state=42)
        X_train2, X_test2, y_train2, y_test2 = load_california_housing_data(random_state=42)
        
        # Check reproducibility
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)
    
    def test_dataset_values_are_reasonable(self):
        """Test that dataset values are within reasonable ranges."""
        X_train, X_test, y_train, y_test = load_california_housing_data()
        
        # Check that features are not all zeros or all the same
        assert not np.all(X_train == 0)
        assert not np.all(X_train == X_train[0, 0])
        
        # Check that targets are positive (housing prices should be positive)
        assert np.all(y_train >= 0)
        assert np.all(y_test >= 0)
        
        # Check that targets are reasonable (housing prices in hundreds of thousands)
        assert np.all(y_train <= 20)  # Max should be around 5-10 (i.e., $500k-$1M)
        assert np.all(y_test <= 20)


class TestModelCreation:
    """Test suite for model creation and validation."""
    
    def test_linear_regression_model_creation(self):
        """Test that LinearRegression model can be created."""
        model = LinearRegression()
        assert isinstance(model, LinearRegression)
    
    def test_model_has_required_attributes_after_training(self):
        """Test that model has required attributes after training."""
        # Load data and train a simple model
        X_train, _, y_train, _ = load_california_housing_data()
        
        # Use a small subset for fast testing
        X_train_small = X_train[:100]
        y_train_small = y_train[:100]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_small)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train_small)
        
        # Check required attributes exist
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        
        # Check attributes have correct shapes
        assert model.coef_.shape == (8,)  # 8 features
        assert isinstance(model.intercept_, (float, np.floating))
    
    def test_validate_model_parameters_function(self):
        """Test the validate_model_parameters utility function."""
        # Train a model
        X_train, _, y_train, _ = load_california_housing_data()
        X_train_small = X_train[:100]
        y_train_small = y_train[:100]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_small)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train_small)
        
        # Test validation function
        assert validate_model_parameters(model) == True
        
        # Test with invalid model (mock object without required attributes)
        class MockModel:
            pass
        
        mock_model = MockModel()
        assert validate_model_parameters(mock_model) == False


class TestModelTraining:
    """Test suite for model training validation."""
    
    def test_model_training_completes_successfully(self):
        """Test that model training completes without errors."""
        # This tests the full training pipeline
        try:
            model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
            assert True  # If we reach here, training succeeded
        except Exception as e:
            pytest.fail(f"Model training failed with error: {str(e)}")
    
    def test_trained_model_can_make_predictions(self):
        """Test that trained model can make predictions."""
        model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
        
        # Test prediction on a small sample
        X_sample = X_test[:5]
        predictions = model.predict(X_sample)
        
        # Check predictions shape
        assert predictions.shape == (5,)
        assert isinstance(predictions[0], (float, np.floating))
        
        # Check predictions are reasonable
        assert np.all(predictions >= 0)  # Housing prices should be positive
        assert np.all(predictions <= 20)  # Should be within reasonable range
    
    def test_model_coefficients_are_reasonable(self):
        """Test that model coefficients are within reasonable ranges."""
        model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
        
        # Check coefficients exist and have reasonable values
        assert len(model.coef_) == 8
        assert not np.any(np.isnan(model.coef_))
        assert not np.any(np.isinf(model.coef_))
        
        # Check intercept
        assert not np.isnan(model.intercept_)
        assert not np.isinf(model.intercept_)
    
    def test_model_files_are_saved(self):
        """Test that model files are saved correctly."""
        # Run training
        train_linear_regression_model()
        
        # Check that files exist
        assert os.path.exists("model.joblib")
        assert os.path.exists("scaler.joblib")
        
        # Test that files can be loaded
        loaded_model = joblib.load("model.joblib")
        loaded_scaler = joblib.load("scaler.joblib")
        
        assert isinstance(loaded_model, LinearRegression)
        assert isinstance(loaded_scaler, StandardScaler)
        
        # Cleanup
        if os.path.exists("model.joblib"):
            os.remove("model.joblib")
        if os.path.exists("scaler.joblib"):
            os.remove("scaler.joblib")


class TestModelPerformance:
    """Test suite for model performance validation."""
    
    def test_r2_score_exceeds_minimum_threshold(self):
        """Test that R² score exceeds minimum threshold."""
        model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
        
        # R² score should be reasonable for linear regression on this dataset
        # California housing dataset typically achieves R² > 0.5 with linear regression
        MIN_R2_THRESHOLD = 0.4  # Conservative threshold
        assert r2 >= MIN_R2_THRESHOLD, f"R² score {r2:.4f} is below minimum threshold {MIN_R2_THRESHOLD}"
        
        # R² score should not be too high (would indicate overfitting or data leakage)
        MAX_R2_THRESHOLD = 0.9
        assert r2 <= MAX_R2_THRESHOLD, f"R² score {r2:.4f} is suspiciously high (>{MAX_R2_THRESHOLD})"
    
    def test_mse_is_reasonable(self):
        """Test that Mean Squared Error is within reasonable bounds."""
        model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
        
        # MSE should be positive
        assert mse > 0, f"MSE should be positive, got {mse}"
        
        # MSE should not be too high (would indicate poor model)
        # For California housing, MSE typically ranges from 0.3 to 1.0
        MAX_MSE_THRESHOLD = 2.0  # Conservative threshold
        assert mse <= MAX_MSE_THRESHOLD, f"MSE {mse:.4f} is too high (>{MAX_MSE_THRESHOLD})"
    
    def test_predictions_vs_actuals_correlation(self):
        """Test that predictions are reasonably correlated with actual values."""
        model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate correlation between predictions and actual values
        correlation = np.corrcoef(y_pred, y_test)[0, 1]
        
        # Correlation should be reasonably high
        MIN_CORRELATION = 0.6  # Conservative threshold
        assert correlation >= MIN_CORRELATION, f"Correlation {correlation:.4f} is too low (<{MIN_CORRELATION})"
        
        # Correlation should not be perfect (would indicate overfitting)
        assert correlation <= 0.99, f"Correlation {correlation:.4f} is suspiciously high"


class TestTrainingRequirements:
    """Test suite for assignment-specific training requirements."""
    
    def test_validate_training_requirements_function(self):
        """Test the validate_training_requirements function."""
        # First train a model
        train_linear_regression_model()
        
        # Test validation function
        try:
            validate_training_requirements()
            assert True  # If we reach here, validation passed
        except Exception as e:
            pytest.fail(f"Training requirements validation failed: {str(e)}")
        
        # Cleanup
        if os.path.exists("model.joblib"):
            os.remove("model.joblib")
        if os.path.exists("scaler.joblib"):
            os.remove("scaler.joblib")
    
    def test_only_linear_regression_is_used(self):
        """Test that only LinearRegression model is used (assignment requirement)."""
        model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
        
        # Check model type
        assert isinstance(model, LinearRegression), f"Expected LinearRegression, got {type(model)}"
        assert type(model).__name__ == "LinearRegression"
        
        # Cleanup
        if os.path.exists("model.joblib"):
            os.remove("model.joblib")
        if os.path.exists("scaler.joblib"):
            os.remove("scaler.joblib")


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_calculate_metrics_function(self):
        """Test the calculate_metrics utility function."""
        # Create simple test data
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
        
        r2, mse = calculate_metrics(y_true, y_pred)
        
        # Check that metrics are reasonable
        assert isinstance(r2, (float, np.floating))
        assert isinstance(mse, (float, np.floating))
        assert r2 >= 0  # R² should be high for this close prediction
        assert mse > 0   # MSE should be positive
        assert mse < 1   # MSE should be small for this close prediction
    
    def test_save_and_load_model_functions(self):
        """Test model saving and loading utility functions."""
        # Create a simple model
        model = LinearRegression()
        X_dummy = np.random.random((10, 3))
        y_dummy = np.random.random(10)
        model.fit(X_dummy, y_dummy)
        
        # Test saving
        test_filepath = "test_model.joblib"
        save_model(model, test_filepath)
        assert os.path.exists(test_filepath)
        
        # Test loading
        loaded_model = load_model(test_filepath)
        assert isinstance(loaded_model, LinearRegression)
        
        # Test that loaded model has same parameters
        np.testing.assert_array_almost_equal(model.coef_, loaded_model.coef_)
        assert abs(model.intercept_ - loaded_model.intercept_) < 1e-10
        
        # Cleanup
        os.remove(test_filepath)


# Pytest fixtures for common test data
@pytest.fixture
def sample_housing_data():
    """Fixture to provide sample housing data for tests."""
    return load_california_housing_data()


@pytest.fixture  
def trained_model():
    """Fixture to provide a trained model for tests."""
    model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
    yield model, scaler, X_test, y_test, r2, mse
    
    # Cleanup after test
    if os.path.exists("model.joblib"):
        os.remove("model.joblib")
    if os.path.exists("scaler.joblib"):
        os.remove("scaler.joblib")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
