#!/bin/bash

# Build Docker image
docker build -t pdvc .

# Run container with GPU support and mounted volumes
docker run --gpus all \
    -v /path/to/features:/app/data/yc2/features \
    -v /path/to/checkpoints:/app/save \
    -it pdvc
