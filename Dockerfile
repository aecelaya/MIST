ARG PYTORCH_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

# Set environment variables for non-interactive installation.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Install MIST. The publish workflow passes the release's version, so a
# pre-release image gets the pre-release (pip skips pre-releases otherwise).
# Left empty, a local build installs the latest stable release.
ARG MIST_VERSION=""
RUN if [ -n "$MIST_VERSION" ]; then \
        pip install --no-cache-dir "mist-medical==${MIST_VERSION}"; \
    else \
        pip install --no-cache-dir mist-medical; \
    fi

# Fail the build if pip installed a different version than asked for.
RUN python -c "import sys; from importlib.metadata import version; v = version('mist-medical'); want = sys.argv[1]; print('mist-medical', v); assert not want or v == want, (v, want)" "$MIST_VERSION"

# PyTorch 2.9's CUDA wheels pin cuDNN 9.10, which runs most 3D convolutions in
# BF16 and FP16 without cuDNN, about 3x slower than FP32 on an H100. cuDNN 9.15+
# fixes it. pip warns that torch pins 9.10; PyTorch works with the newer cuDNN.
RUN pip install --no-cache-dir -U "nvidia-cudnn-cu12>=9.15"

# Fail the build if PyTorch still loads the old cuDNN. This needs no GPU.
RUN python -c "import torch; v = torch.backends.cudnn.version(); print('cuDNN', v); assert v and v >= 91500, v"

# Create app directory.
RUN mkdir /app
WORKDIR /app