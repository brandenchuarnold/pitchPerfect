# docker/Dockerfile
FROM python:3.9-slim

# Install system dependencies for OCR, spaCy, etc.
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    wget \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Add apt-get instructions to install libGL
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/requirements.txt requirements.txt

# Install compatible numpy and other dependencies from requirements.txt
RUN pip install --no-cache-dir numpy==1.21.6
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

# Download spaCy English model
RUN python -m spacy download en_core_web_sm

# Start ADB server
RUN adb kill-server
RUN adb start-server

EXPOSE 8000

CMD ["python", "main.py"]
