import os
import sys
import pytest
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import (
    load_california_housing_data,
    train_linear_regression_model,
    validate_training_requirements
)

### Minimal re-implementations of utility functions for self-contained testing ###
def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import r2_score, mean_squared_error
    return r2_score(y_true, y_pred), mean_squared_error(y_true, y_pred)

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)

def validate_model_parameters(model):
    try:
        has_coef = hasattr(model, 'coef_')
        has_intercept = hasattr(model, 'intercept_')
        return has_coef and has_intercept
    except Exception:
        return False

###############################################################################

class TestDatasetLoading:
    def test_load_california_housing_data_returns_correct_shapes(self):
        X_train, X_test, y_train, y_test = load_california_housing_data()
        assert len([X_train, X_test, y_train, y_test]) == 4
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert X_train.shape[1] == X_test.shape[1]
        assert X_train.shape[1] == 8
        assert X_test.shape[1] == 8

    def test_load_california_housing_data_with_custom_test_size(self):
        test_size = 0.3
        X_train, X_test, y_train, y_test = load_california_housing_data(test_size=test_size)
        total_samples = X_train.shape[0] + X_test.shape[0]
        actual_test_ratio = X_test.shape[0] / total_samples
        assert abs(actual_test_ratio - test_size) < 0.01

    def test_load_california_housing_data_reproducibility(self):
        X_train1, X_test1, y_train1, y_test1 = load_california_housing_data(random_state=42)
        X_train2, X_test2, y_train2, y_test2 = load_california_housing_data(random_state=42)
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)

    def test_dataset_values_are_reasonable(self):
        X_train, X_test, y_train, y_test = load_california_housing_data()
        assert not np.all(X_train == 0)
        assert not np.all(X_train == X_train[0, 0])
        assert np.all(y_train >= 0)
        assert np.all(y_test >= 0)
        assert np.all(y_train <= 20)
        assert np.all(y_test <= 20)

class TestModelCreation:
    def test_linear_regression_model_creation(self):
        model = LinearRegression()
        assert isinstance(model, LinearRegression)

    def test_model_has_required_attributes_after_training(self):
        X_train, _, y_train, _ = load_california_housing_data()
        X_train_small = X_train[:100]
        y_train_small = y_train[:100]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_small)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train_small)
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_.shape == (8,)
        assert isinstance(model.intercept_, (float, np.floating))

    def test_validate_model_parameters_function(self):
        X_train, _, y_train, _ = load_california_housing_data()
        X_train_small = X_train[:100]
        y_train_small = y_train[:100]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_small)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train_small)
        assert validate_model_parameters(model) == True
        class MockModel:
            pass
        mock_model = MockModel()
        assert validate_model_parameters(mock_model) == False

class TestModelTraining:
    def test_model_training_completes_successfully(self):
        try:
            model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
            assert True
        except Exception as e:
            pytest.fail(f"Model training failed with error: {str(e)}")

    def test_trained_model_can_make_predictions(self):
        model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
        X_sample = X_test[:5]
        predictions = model.predict(X_sample)
        assert predictions.shape == (5,)
        assert isinstance(predictions[0], (float, np.floating))
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 20)

    def test_model_coefficients_are_reasonable(self):
        model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
        assert len(model.coef_) == 8
        assert not np.any(np.isnan(model.coef_))
        assert not np.any(np.isinf(model.coef_))
        assert not np.isnan(model.intercept_)
        assert not np.isinf(model.intercept_)

    def test_model_files_are_saved(self):
        train_linear_regression_model()
        assert os.path.exists("model.joblib")
        assert os.path.exists("scaler.joblib")
        loaded_model = joblib.load("model.joblib")
        loaded_scaler = joblib.load("scaler.joblib")
        assert isinstance(loaded_model, LinearRegression)
        assert isinstance(loaded_scaler, StandardScaler)
        if os.path.exists("model.joblib"):
            os.remove("model.joblib")
        if os.path.exists("scaler.joblib"):
            os.remove("scaler.joblib")

class TestModelPerformance:
    def test_r2_score_exceeds_minimum_threshold(self):
        model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
        MIN_R2_THRESHOLD = 0.4
        assert r2 >= MIN_R2_THRESHOLD, f"R² score {r2:.4f} is below minimum threshold {MIN_R2_THRESHOLD}"
        MAX_R2_THRESHOLD = 0.9
        assert r2 <= MAX_R2_THRESHOLD, f"R² score {r2:.4f} is suspiciously high (>{MAX_R2_THRESHOLD})"

    def test_mse_is_reasonable(self):
        model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
        assert mse > 0, f"MSE should be positive, got {mse}"
        MAX_MSE_THRESHOLD = 2.0
        assert mse <= MAX_MSE_THRESHOLD, f"MSE {mse:.4f} is too high (>{MAX_MSE_THRESHOLD})"

    def test_predictions_vs_actuals_correlation(self):
        model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
        y_pred = model.predict(X_test)
        correlation = np.corrcoef(y_pred, y_test)[0, 1]
        MIN_CORRELATION = 0.6
        assert correlation >= MIN_CORRELATION, f"Correlation {correlation:.4f} is too low (<{MIN_CORRELATION})"
        assert correlation <= 0.99, f"Correlation {correlation:.4f} is suspiciously high"

class TestTrainingRequirements:
    def test_validate_training_requirements_function(self):
        train_linear_regression_model()
        try:
            validate_training_requirements()
            assert True
        except Exception as e:
            pytest.fail(f"Training requirements validation failed: {str(e)}")
        if os.path.exists("model.joblib"):
            os.remove("model.joblib")
        if os.path.exists("scaler.joblib"):
            os.remove("scaler.joblib")

    def test_only_linear_regression_is_used(self):
        model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
        assert isinstance(model, LinearRegression), f"Expected LinearRegression, got {type(model)}"
        assert type(model).__name__ == "LinearRegression"
        if os.path.exists("model.joblib"):
            os.remove("model.joblib")
        if os.path.exists("scaler.joblib"):
            os.remove("scaler.joblib")

class TestUtilityFunctions:
    def test_calculate_metrics_function(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
        r2, mse = calculate_metrics(y_true, y_pred)
        assert isinstance(r2, (float, np.floating))
        assert isinstance(mse, (float, np.floating))
        assert r2 >= 0
        assert mse > 0
        assert mse < 1

    def test_save_and_load_model_functions(self):
        model = LinearRegression()
        X_dummy = np.random.random((10, 3))
        y_dummy = np.random.random(10)
        model.fit(X_dummy, y_dummy)
        test_filepath = "test_model.joblib"
        save_model(model, test_filepath)
        assert os.path.exists(test_filepath)
        loaded_model = load_model(test_filepath)
        assert isinstance(loaded_model, LinearRegression)
        np.testing.assert_array_almost_equal(model.coef_, loaded_model.coef_)
        assert abs(model.intercept_ - loaded_model.intercept_) < 1e-10
        os.remove(test_filepath)

@pytest.fixture
def sample_housing_data():
    return load_california_housing_data()

@pytest.fixture
def trained_model():
    model, scaler, X_test, y_test, r2, mse = train_linear_regression_model()
    yield model, scaler, X_test, y_test, r2, mse
    if os.path.exists("model.joblib"):
        os.remove("model.joblib")
    if os.path.exists("scaler.joblib"):
        os.remove("scaler.joblib")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
