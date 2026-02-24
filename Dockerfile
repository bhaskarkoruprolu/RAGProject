# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Configure pip with high timeout and retries
RUN pip config set global.timeout 300 && \
    pip config set global.retries 10

# Upgrade pip and setuptools first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install system dependencies needed for native packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY Requirements.txt .

# Step 1: Install PyTorch CPU-only first from PyTorch's own CDN
# Using 2.4.0+cpu to satisfy sentence-transformers requirements
RUN pip install --no-cache-dir \
    torch==2.4.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install sentence-transformers (now that torch is present)
RUN pip install --no-cache-dir sentence-transformers

# Step 3: Install lighter packages
RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    python-dotenv \
    pypdf \
    pymupdf \
    typesense \
    docx2txt \
    unstructured \
    openpyxl \
    jq


# Step 4: Install langchain stack
RUN pip install --no-cache-dir \
    langchain \
    langchain-core \
    langchain-community \
    langchain-text-splitters \
    langchain-groq \
    langchain_openai \
    langgraph

# Step 5: Install faiss and chromadb
RUN pip install --no-cache-dir \
    faiss-cpu \
    chromadb

# Copy the rest of the application code
COPY . .

# Expose port 5000 for the Flask app
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=server.py

# Run the application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"]
