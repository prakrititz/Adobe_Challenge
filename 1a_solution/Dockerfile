# Use Python 3.9 slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including build tools for fasttext
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the fastText language model
RUN wget -O /app/lid.176.ftz https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

# Copy the main script
COPY pdf_extractor.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set the script as executable
RUN chmod +x pdf_extractor.py

# Run the PDF extractor when container starts
CMD ["python", "pdf_extractor.py"]