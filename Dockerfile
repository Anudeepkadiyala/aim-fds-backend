# Use official Python image
FROM python:3.10-slim

# ✅ Install OpenCV-related system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port used by the app
EXPOSE 8080

# Start the FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
