# docker/Dockerfile
FROM python:3.9-slim

# Set non-interactive frontend to prevent prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OCR, spaCy, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libssl-dev \
    wget \
    tesseract-ocr \
    libtesseract-dev \
    libgl1-mesa-glx \
    adb \
    && rm -rf /var/lib/apt/lists/* \
    && echo "System dependencies installed"

# Make work directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY app/requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --progress-bar on numpy==1.21.6 && \
    pip install --no-cache-dir --progress-bar on -r requirements.txt && \
    echo "Python dependencies installed"

# Download spaCy English model
RUN python -m spacy download en_core_web_sm && \
    echo "spaCy model downloaded"

# Copy application code
COPY app/ /app/
COPY .env .env

EXPOSE 8000

CMD ["bash", "-c", "adb start-server && python main.py"]