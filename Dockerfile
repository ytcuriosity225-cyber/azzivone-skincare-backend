# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Cloud Run expects the container to listen on the port defined by the PORT env var (usually 8080)
ENV PORT=8080

# Install essential system dependencies
# libgl1 is often required for image processing libraries (even if using Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# This includes main.py, azzivone_model.tflite, and the Excel file
COPY . .

# Expose the port for documentation (Cloud Run handles the actual mapping)
EXPOSE 8080

# Run the web service on container startup.
# We use uvicorn to serve the FastAPI app. 
# We bind to 0.0.0.0 and use the PORT environment variable provided by Cloud Run.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
