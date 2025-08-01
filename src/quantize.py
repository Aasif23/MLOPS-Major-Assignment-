import os
import numpy as np
import joblib
import logging
from typing import Tuple, Dict, Any

from utils import load_model, save_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_model_parameters(model: Any) -> Dict[str, np.ndarray]:
    """
    Extract coefficients and intercept from trained Linear Regression model.
    
    Args:
        model: Trained LinearRegression model
        
    Returns:
        Dictionary containing 'coef' and 'intercept' parameters
    """
    logger.info("Extracting model parameters...")
    
    if not hasattr(model, 'coef_') or not hasattr(model, 'intercept_'):
        raise AttributeError("Model missing required parameters (coef_ or intercept_)")
    
    parameters = {
        'coef': model.coef_.copy(),
        'intercept': np.array([model.intercept_])  # Make it an array for consistency
    }
    
    logger.info(f"Extracted coefficients shape: {parameters['coef'].shape}")
    logger.info(f"Extracted intercept shape: {parameters['intercept'].shape}")
    
    return parameters


def quantize_to_uint8(values: np.ndarray, min_val: float = None, max_val: float = None) -> Tuple[np.ndarray, float, float]:
    """
    Quantize floating point values to 8-bit unsigned integers (0-255).
    
    Args:
        values: Input floating point values
        min_val: Minimum value for quantization range (optional)
        max_val: Maximum value for quantization range (optional)
        
    Returns:
        Tuple of (quantized_values, scale, zero_point)
    """
    # Determine the range of values
    if min_val is None:
        min_val = float(np.min(values))
    if max_val is None:
        max_val = float(np.max(values))
    
    # Ensure we have a valid range
    if min_val == max_val:
        # Handle edge case where all values are the same
        scale = 1.0
        zero_point = 128.0  # Middle of uint8 range
        quantized = np.full_like(values, 128, dtype=np.uint8)
    else:
        # Calculate scale and zero point for symmetric quantization
        # Map floating point range to [0, 255]
        scale = (max_val - min_val) / 255.0
        zero_point = -min_val / scale
        
        # Clip zero_point to valid range
        zero_point = np.clip(zero_point, 0, 255)
        
        # Quantize values
        quantized_float = (values - min_val) / scale
        quantized = np.clip(np.round(quantized_float), 0, 255).astype(np.uint8)
    
    logger.info(f"Quantization - Min: {min_val:.6f}, Max: {max_val:.6f}")
    logger.info(f"Scale: {scale:.6f}, Zero point: {zero_point:.6f}")
    
    return quantized, scale, zero_point


def dequantize_from_uint8(quantized_values: np.ndarray, scale: float, zero_point: float, min_val: float) -> np.ndarray:
    """
    Dequantize 8-bit unsigned integers back to floating point values.
    
    Args:
        quantized_values: Quantized uint8 values
        scale: Scale factor used in quantization
        zero_point: Zero point used in quantization
        min_val: Original minimum value
        
    Returns:
        Dequantized floating point values
    """
    # Convert back to floating point
    dequantized = quantized_values.astype(np.float32) * scale + min_val
    
    return dequantized


def quantize_model_parameters(parameters: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Quantize all model parameters to 8-bit unsigned integers.
    
    Args:
        parameters: Dictionary containing model parameters
        
    Returns:
        Dictionary containing quantized parameters and metadata
    """
    logger.info("Starting model parameter quantization...")
    
    quantized_params = {}
    
    for param_name, param_values in parameters.items():
        logger.info(f"Quantizing parameter: {param_name}")
        
        # Quantize the parameter
        quant_values, scale, zero_point = quantize_to_uint8(param_values)
        
        # Store quantized data and metadata
        quantized_params[param_name] = {
            'quantized_values': quant_values,
            'scale': scale,
            'zero_point': zero_point,
            'original_shape': param_values.shape,
            'min_val': float(np.min(param_values)),
            'max_val': float(np.max(param_values))
        }
        
        # Verify quantization by dequantizing
        dequant_values = dequantize_from_uint8(
            quant_values, scale, zero_point, quantized_params[param_name]['min_val']
        )
        
        # Calculate quantization error
        error = np.mean(np.abs(param_values - dequant_values))
        max_error = np.max(np.abs(param_values - dequant_values))
        
        logger.info(f"Quantization error - Mean: {error:.6f}, Max: {max_error:.6f}")
        
        print(f"Parameter: {param_name}")
        print(f"  Original shape: {param_values.shape}")
        print(f"  Original range: [{quantized_params[param_name]['min_val']:.6f}, {quantized_params[param_name]['max_val']:.6f}]")
        print(f"  Quantized range: [0, 255] (uint8)")
        print(f"  Scale: {scale:.6f}")
        print(f"  Zero point: {zero_point:.6f}")
        print(f"  Mean quantization error: {error:.6f}")
        print(f"  Max quantization error: {max_error:.6f}")
        print("-" * 50)
    
    return quantized_params


def perform_quantized_inference(quantized_params: Dict[str, Any], X_sample: np.ndarray) -> np.ndarray:
    """
    Perform inference using quantized parameters.
    
    Args:
        quantized_params: Quantized model parameters
        X_sample: Input features for prediction
        
    Returns:
        Predictions using dequantized parameters
    """
    logger.info("Performing inference with quantized parameters...")
    
    # Dequantize coefficients
    coef_data = quantized_params['coef']
    dequant_coef = dequantize_from_uint8(
        coef_data['quantized_values'],
        coef_data['scale'],
        coef_data['zero_point'],
        coef_data['min_val']
    )
    
    # Dequantize intercept
    intercept_data = quantized_params['intercept']
    dequant_intercept = dequantize_from_uint8(
        intercept_data['quantized_values'],
        intercept_data['scale'],
        intercept_data['zero_point'],
        intercept_data['min_val']
    )[0]  # Extract scalar value
    
    # Perform linear regression prediction: y = X * coef + intercept
    predictions = X_sample @ dequant_coef + dequant_intercept
    
    return predictions


def main():
    """
    Main quantization workflow.
    """
    try:
        # Check if trained model exists
        model_path = "model.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Please run training first.")
        
        # Load trained model
        logger.info("Loading trained model...")
        model = load_model(model_path)
        
        # Extract model parameters
        raw_params = extract_model_parameters(model)
        
        # Save raw parameters
        raw_params_path = "unquant_params.joblib"
        save_model(raw_params, raw_params_path)
        logger.info(f"Raw parameters saved to {raw_params_path}")
        
        # Quantize parameters
        quantized_params = quantize_model_parameters(raw_params)
        
        # Save quantized parameters
        quant_params_path = "quant_params.joblib"
        save_model(quantized_params, quant_params_path)
        logger.info(f"Quantized parameters saved to {quant_params_path}")
        
        # Test quantized inference with sample data
        logger.info("Testing quantized inference...")
        
        # Load test data (use same data loading as training)
        from utils import load_california_housing_data
        _, X_test, _, y_test = load_california_housing_data()
        
        # Load scaler if it exists
        scaler_path = "scaler.joblib"
        if os.path.exists(scaler_path):
            scaler = load_model(scaler_path)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Use a small sample for testing
        X_sample = X_test_scaled[:5]
        y_sample = y_test[:5]
        
        # Original model predictions
        original_pred = model.predict(X_sample)
        
        # Quantized model predictions
        quantized_pred = perform_quantized_inference(quantized_params, X_sample)
        
        # Compare predictions
        pred_error = np.mean(np.abs(original_pred - quantized_pred))
        max_pred_error = np.max(np.abs(original_pred - quantized_pred))
        
        print("\\n" + "="*60)
        print("QUANTIZATION INFERENCE TEST")
        print("="*60)
        print("Sample predictions comparison:")
        for i in range(len(X_sample)):
            print(f"  Sample {i+1}: Original={original_pred[i]:.4f}, Quantized={quantized_pred[i]:.4f}, True={y_sample[i]:.4f}")
        print(f"\\nPrediction Error Statistics:")
        print(f"  Mean error: {pred_error:.6f}")
        print(f"  Max error: {max_pred_error:.6f}")
        print("="*60)
        
        # Success message
        print("\\n✅ Quantization completed successfully!")
        print(f"📁 Raw parameters saved: {raw_params_path}")
        print(f"🔢 Quantized parameters saved: {quant_params_path}")
        print(f"📊 Mean prediction error: {pred_error:.6f}")
        
    except Exception as e:
        logger.error(f"Quantization failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
