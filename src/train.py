from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import joblib
import os

def load_california_housing_data(test_size=0.2, random_state=42):
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def train_linear_regression_model():
    # Load data and preprocess
    X_train, X_test, y_train, y_test = load_california_housing_data()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    # Save artifacts
    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    return model, scaler, X_test_scaled, y_test, r2, mse

def validate_training_requirements():
    """
    Checks for assignment compliance:
    - Only LinearRegression/8 features/model and scaler files exist
    - Model and scaler are loadable
    """
    # Check model type and files
    assert os.path.exists("model.joblib"), "model.joblib not found."
    assert os.path.exists("scaler.joblib"), "scaler.joblib not found."
    model = joblib.load("model.joblib")
    assert isinstance(model, LinearRegression), "Only LinearRegression allowed!"
    scaler = joblib.load("scaler.joblib")
    X_train, _, y_train, _ = load_california_housing_data()
    assert X_train.shape[1] == 8, "Feature count is NOT 8!"
    assert scaler.mean_.shape[0] == 8, "Scaler fitted on wrong feature count."

def main():
    X_train, X_test, y_train, y_test = load_california_housing_data()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    print(f"Training R² Score: {r2_score(y_train, train_pred):.4f}")
    print(f"Training MSE (loss): {mean_squared_error(y_train, train_pred):.4f}")
    print(f"Test R² Score: {r2_score(y_test, test_pred):.4f}")
    print(f"Test MSE (loss): {mean_squared_error(y_test, test_pred):.4f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, test_pred)):.4f}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Model intercept: {model.intercept_:.4f}")
    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")

if __name__ == "__main__":
    main()
