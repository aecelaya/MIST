ARG PYTORCH_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

# Set environment variables for non-interactive installation.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Install MIST.
RUN pip install --no-cache-dir "mist-medical[train-cuda]"

# Create app directory.
RUN mkdir /app
WORKDIR /app