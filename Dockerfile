# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set maintainer
LABEL maintainer="your-email@domain.com"
LABEL description="911 Emergency Calls Analytics Platform"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models plots reports logs

# Set permissions
RUN chmod +x scripts/run_pipeline.py

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

# Default command - run dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Alternative commands available:
# For data processing: docker run <image> python scripts/run_pipeline.py --data-only
# For model training: docker run <image> python scripts/run_pipeline.py --train-only --data-path data/processed/processed_data.parquet
# For full pipeline: docker run <image> python scripts/run_pipeline.py --full 