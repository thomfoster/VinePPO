# Use the official PyTorch image as the base image
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV TZ=America/New_York
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

# Update the package list and install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    curl \
    unzip \
    software-properties-common \
    apt-transport-https ca-certificates gnupg lsb-release gnupg \
    make bash-completion tree vim less tmux nmap nano wget \
    build-essential libfreetype6-dev \
    pkg-config \
    graphviz \
    git \
    htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create local user
# https://jtreminio.com/blog/running-docker-containers-as-current-host-user/
ARG UID
ARG GID
RUN if [ ${UID:-0} -ne 0 ] && [ ${GID:-0} -ne 0 ]; then \
    groupadd -g ${GID} duser &&\
    useradd -l -u ${UID} -g duser duser &&\
    install -d -m 0755 -o duser -g duser /home/duser &&\
    chown --changes --silent --no-dereference --recursive ${UID}:${GID} /home/duser \
    ;fi

USER duser
WORKDIR /home/duser

# Install Python packages
ENV PATH="/home/duser/.local/bin:$PATH"
RUN python3 -m pip install --upgrade pip
ARG REQS
RUN pip install $REQS

WORKDIR /home/duser/project


