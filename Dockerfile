# Use CUDA 11.7 compatible base image
FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.9 \
    python3.9-dev \
    python3-pip \
    ffmpeg \
    gcc \
    g++ \
    make \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Clone the repository
RUN git clone --recursive https://github.com/trtuananh/Dense-Video-Captioning.git .

# Install Python dependencies
RUN python3.9 -m pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1
COPY requirement.txt .
RUN python3.9 -m pip install --no-cache-dir -r requirement.txt

# Compile deformable attention layer
RUN cd pdvc/ops && sh make.sh

# Create directories for features and checkpoints
RUN mkdir -p data/yc2/features \
    && mkdir -p save

# Instructions for manual steps (as comments)
# Note: Due to large file sizes, these steps need to be done manually:
# 1. Download tsp_mvitv2 features to data/yc2/features
# 2. Download sound_feature_train to data/yc2/features
# 3. Download checkpoints to save/

# Set default command to show instructions
CMD ["echo", "Container is ready. Please mount feature and checkpoint volumes when running the container."]
