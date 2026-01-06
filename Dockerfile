# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:25.03-py3

# Avoid interactive prompts, keep logs clean, and make Python behave nicely in containers
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# (Optional but recommended) Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Copy only requirements first to maximize layer caching:
# If requirements.txt changes, this layer invalidates and rebuilds.
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Download spaCy model at build time so runtime startup is fast/offline-friendly
RUN python -m spacy download en_core_web_sm

# Create standard mount points (not strictly required, but makes intent clear)
RUN mkdir -p /app/src /app/data /app/out

# Default command runs your entrypoint. Source will be mounted at runtime.
CMD ["python", "/app/src/main.py"]
