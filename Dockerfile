# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Start FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]

