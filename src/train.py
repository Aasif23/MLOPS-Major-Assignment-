from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import joblib

def load_data():
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_data()

    # Use original features only; no polynomial expansion
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

    # Save model and scaler
    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")

if __name__ == "__main__":
    main()
