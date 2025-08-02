import numpy as np
import joblib

def quantize_to_uint8(arr: np.ndarray):
    min_val = arr.min()
    max_val = arr.max()
    if max_val == min_val:
        scale = 1.0
    else:
        scale = (max_val - min_val) / 255
    zero_point = np.round(-min_val / scale).astype(np.uint8) if scale != 0 else 0
    quantized = np.clip(np.round(arr / scale + zero_point), 0, 255).astype(np.uint8)
    return quantized, scale, zero_point

def dequantize_param(quant_param: dict):
    if quant_param["scale"] == 1.0 and quant_param["zero_point"] == 0:
        # Intercept stored as float32 directly; return as is
        return quant_param["quantized_values"].reshape(quant_param["original_shape"])
    else:
        quant_values = quant_param["quantized_values"].astype(np.float32)
        return (quant_values - quant_param["zero_point"]) * quant_param["scale"]

def quantize_model_parameters(model):
    parameters = {
        "coef": model.coef_,
        "intercept": np.array([model.intercept_]) if np.ndim(model.intercept_) == 0 else np.array(model.intercept_)
    }
    quantized_params = {}
    for name, values in parameters.items():
        if name == "intercept":
            quantized_params[name] = {
                "quantized_values": values.astype(np.float32),
                "scale": 1.0,
                "zero_point": 0,
                "original_shape": values.shape,
                "min_val": float(values.min()),
                "max_val": float(values.max())
            }
        else:
            quant_values, scale, zero_point = quantize_to_uint8(values)
            quantized_params[name] = {
                "quantized_values": quant_values,
                "scale": scale,
                "zero_point": zero_point,
                "original_shape": values.shape,
                "min_val": float(values.min()),
                "max_val": float(values.max())
            }
    return quantized_params

def save_quantized_params(quantized_params, filename="quant_params.joblib"):
    joblib.dump(quantized_params, filename)
    print(f"Quantized parameters saved to {filename}")

def load_quantized_params(filename="quant_params.joblib"):
    return joblib.load(filename)

def predict_with_quantized_params(X, quant_params):
    coef = dequantize_param(quant_params["coef"]).flatten()
    intercept = dequantize_param(quant_params["intercept"]).flatten()
    return np.dot(X, coef) + intercept

def calculate_quantization_error(original_params, quant_params):
    errors = {}
    for key in ["coef", "intercept"]:
        orig = original_params[key]
        dequant = dequantize_param(quant_params[key])
        errors[key] = (float(np.abs(orig - dequant).mean()), float(np.abs(orig - dequant).max()))
    return errors

def main():
    print("Loading California Housing data...")
    from sklearn.datasets import fetch_california_housing
    cali = fetch_california_housing()
    X, y = cali.data, cali.target

    print("Loading scaler and model...")
    scaler = joblib.load("scaler.joblib")
    model = joblib.load("model.joblib")

    X_scaled = scaler.transform(X)

    original_params = {
        "coef": model.coef_,
        "intercept": np.array([model.intercept_]) if np.ndim(model.intercept_) == 0 else np.array(model.intercept_)
    }

    quantized_params = quantize_model_parameters(model)
    save_quantized_params(quantized_params)

    error_stats = calculate_quantization_error(original_params, quantized_params)
    print(f"Quantization error (mean abs) - coef: {error_stats['coef'][0]:.6f}, intercept: {error_stats['intercept'][0]:.6f}")
    print(f"Quantization error (max abs) - coef: {error_stats['coef'][1]:.6f}, intercept: {error_stats['intercept'][1]:.6f}")

    original_pred = model.predict(X_scaled)
    quant_pred = predict_with_quantized_params(X_scaled, quantized_params)
    pred_error = np.abs(original_pred - quant_pred)

    print(f"Prediction Error Statistics:")
    print(f"  Mean error: {pred_error.mean():.6f}")
    print(f"  Max error: {pred_error.max():.6f}")

if __name__ == "__main__":
    main()
