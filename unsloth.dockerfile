FROM nvidia/cuda:12.6.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and pip
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3-pip \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure python3 points to Python 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip (optional but recommended)
RUN python3 -m pip install --upgrade pip

# Install required Python packages
RUN pip install torch==2.7.0 torchvision==0.22.0 unsloth==2025.4.7
RUN pip install flwr==1.18.0

WORKDIR /workspace

COPY finetune.py .

CMD ["python3","finetune.py"]
