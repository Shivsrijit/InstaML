FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

# Add PyTorch CPU wheels first
RUN pip install --upgrade pip
RUN pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cpu

# Install rest of dependencies
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Expose streamlit port
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
