FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy backend and frontend requirements
COPY backend/requirements.txt ./backend_requirements.txt
COPY frontend/requirements.txt ./frontend_requirements.txt

# Install all dependencies
RUN pip install --no-cache-dir -r backend_requirements.txt
RUN pip install --no-cache-dir -r frontend_requirements.txt

# Copy the entire project
COPY . .

# Create data directories
RUN mkdir -p /app/backend/data/repos /app/backend/data/graphs /app/backend/data/chroma

# Make start script executable
RUN chmod +x start.sh

# Expose Hugging Face default port
EXPOSE 7860

# Start script
CMD ["bash", "start.sh"]
