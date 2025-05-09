# Use official Python image (full version)
FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.docker.txt .

# Install dependencies
RUN pip install --no-cache-dir streamlit==1.32.0 && \
    pip install --no-cache-dir pandas==2.2.3 && \
    pip install --no-cache-dir numpy==2.2.5 && \
    pip install --no-cache-dir matplotlib==3.10.1 && \
    pip install --no-cache-dir seaborn==0.13.2 && \
    pip install --no-cache-dir scikit-learn==1.6.1 && \
    pip install --no-cache-dir xgboost==3.0.0 && \
    pip install --no-cache-dir SQLAlchemy==2.0.40

# Copy application code
COPY . .

# Create directory for SQLite database
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port for Streamlit
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"] 