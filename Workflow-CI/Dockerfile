FROM python:3.12.7-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY MLProject/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY MLProject/ .

# Expose port for metrics
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLflow_TRACKING_URI=http://localhost:5000

# Default command
CMD ["python", "modelling_ci.py"]
