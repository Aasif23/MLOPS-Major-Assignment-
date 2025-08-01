# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and the saved model
COPY src/predict.py ./src/
RUN touch src/__init__.py
COPY src/model.joblib .
COPY src/scaler.joblib .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash mluser
RUN chown -R mluser:mluser /app
USER mluser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD python -c "import joblib; joblib.load('model.joblib')" || exit 1

CMD ["python", "src/predict.py"]
