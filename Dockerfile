FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set utility env varibles
ENV PATH=/text_auto_classification_private/miniconda/bin:$PATH

# Set paths as env variables
ARG DEFAULT_SERVICE_CONFIG
ARG DEFAULT_TRAINING_CONFIG

ENV DEFAULT_SERVICE_CONFIG=${DEFAULT_SERVICE_CONFIG}
ENV DEFAULT_TRAINING_CONFIG=${DEFAULT_TRAINING_CONFIG}

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    build-essential \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /text_auto_classification_private

# Install Miniconda and Python
RUN curl -sLo /text_auto_classification_private/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py311_24.1.2-0-Linux-x86_64.sh \
 && chmod +x /text_auto_classification_private/miniconda.sh \
 && /text_auto_classification_private/miniconda.sh -b -p /text_auto_classification_private/miniconda \
 && rm /text_auto_classification_private/miniconda.sh \
 && conda install -y python==3.11 \
 && pip3 install nvitop

# Install python requirements
COPY text_auto_classification/requirements.txt .
RUN pip3 install -r requirements.txt --no-cache

# Copy code to container
COPY text_auto_classification/ text_auto_classification/
COPY etc/ etc/
COPY version.txt .

EXPOSE 8080
CMD uvicorn --host 0.0.0.0 --port 8080 text_auto_classification.fastapi_app:app
