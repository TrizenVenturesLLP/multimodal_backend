# Use full Python image (more stable for cloud builds and includes git/build-essential)
FROM python:3.11-bookworm

# Install remaining system dependencies (ffmpeg and GL libraries)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8010

# Start command
CMD uvicorn multimodal_main:app --host 0.0.0.0 --port ${PORT:-8010}
