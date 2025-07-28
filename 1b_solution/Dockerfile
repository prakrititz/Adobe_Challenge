# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PyMuPDF and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script
COPY pdf_analyser.py .

# Create a directory for the Llama model (optional, if you want to include it in the image)
# COPY reasoning-llama3.2-1b.Q6_K.gguf .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the PDF analyzer
CMD ["python", "pdf_analyser.py"]