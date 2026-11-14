ARG PYTORCH_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

# Set environment variables for non-interactive installation.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Install MIST.
RUN pip install --no-cache-dir mist-medical

# PyTorch 2.9's CUDA wheels pin cuDNN 9.10, which runs most 3D convolutions in
# BF16 and FP16 without cuDNN, about 3x slower than FP32 on an H100. cuDNN 9.15+
# fixes it. pip warns that torch pins 9.10; PyTorch works with the newer cuDNN.
RUN pip install --no-cache-dir -U "nvidia-cudnn-cu12>=9.15"

# Create app directory.
RUN mkdir /app
WORKDIR /app