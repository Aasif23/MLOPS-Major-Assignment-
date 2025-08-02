# MLOps Assignment - California Housing Linear Regression Pipeline

This repository contains a complete MLOps pipeline for Linear Regression using the California Housing dataset from sklearn. The pipeline includes training, testing, quantization, Dockerization, and CI/CD automation.

## Project Structure

```
project-root/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI/CD workflow
├── src/
│   ├── __init__.py
│   ├── train.py                   # Model training script
│   ├── quantize.py                # Model quantization script
│   ├── predict.py                 # Prediction script for Docker
│   └── utils.py                   # Utility functions
├── tests/
│   ├── __init__.py
│   └── test_train.py              # Unit tests
├── Dockerfile                     # Docker configuration
├── .gitignore                     # Git ignore file
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Features

- **Model Training**: Linear Regression on California Housing dataset
- **Model Testing**: Comprehensive unit tests with pytest
- **Model Quantization**: Manual 8-bit quantization of model parameters
- **Containerization**: Docker setup for model deployment
- **CI/CD Pipeline**: Automated testing, training, and deployment with GitHub Actions

## Requirements

- Python 3.8+
- scikit-learn
- numpy
- joblib
- pytest

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training the Model

```bash
python src/train.py
```

This will:
- Load the California Housing dataset
- Train a Linear Regression model
- Print R² score and loss metrics
- Save the trained model as `model.joblib`
<img width="626" height="148" alt="image" src="https://github.com/user-attachments/assets/6941673d-05e8-41fc-b6e7-1fd2fa1d9370" />

### 3. Quantizing the Model

```bash
python src/quantize.py
```

This will:
- Load the trained model
- Extract coefficients and intercept
- Save raw parameters as `unquant_params.joblib`
- Quantize parameters to 8-bit unsigned integers
- Save quantized parameters as `quant_params.joblib`
- Perform inference with de-quantized weights
<img width="638" height="191" alt="image" src="https://github.com/user-attachments/assets/dfdca6eb-512a-4c63-9cd7-802f18a3bc88" />

### 4. Running Tests

```bash
pytest tests/ -v
```
<img width="1607" height="431" alt="image" src="https://github.com/user-attachments/assets/5f1f01bc-9132-421b-98ca-ae5ded7fd1ab" />

### 5. Docker Deployment

```bash
# Build Docker image
docker build -t ml-model .

# Run Docker container
docker run ml-model
```
<img width="1485" height="682" alt="image" src="https://github.com/user-attachments/assets/2c2df5fd-a82f-40dc-b45f-61b8f2ef931f" />


### 🐳 DockerHub Image

You can pull the latest Docker image from [DockerHub](https://hub.docker.com/r/aasifali231/ml-model/tags):

```bash
docker pull aasifali231/ml-model
docker run --rm aasifali231/ml-model
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) includes three jobs:

1. **test_suite**: Runs pytest tests
2. **train_and_quantize**: Trains model and runs quantization
3. **build_and_test_container**: Builds Docker image and tests container

The pipeline is triggered on every push to the main branch.

## Model Performance

| Metric | Value |
|--------|-------|
| Model Type | Linear Regression |
| Dataset | California Housing (sklearn) |
| R² Score | ~0.60 (typical) |
| Features | 8 numerical features |
| Target | Median house value |

## Quantization Details

- **Precision**: 8-bit unsigned integers (0-255)
- **Parameters Quantized**: Model coefficients and intercept
- **Format**: Manual quantization with scale and zero-point
- **Storage**: Separate files for original and quantized parameters

## Docker Configuration

The Docker container:
- Uses Python 3.9 slim base image
- Installs all required dependencies
- Includes the trained model and prediction script
- Runs `predict.py` on container start

## Testing Strategy

Unit tests cover:
- Dataset loading functionality
- Model creation and validation
- Training process verification
- R² score threshold validation
- Parameter extraction for quantization


## License

This project is for educational purposes as part of an MLOps assignment.

## Contact

For questions about this assignment, please contact the course instructors.
