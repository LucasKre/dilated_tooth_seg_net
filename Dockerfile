FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set bash as the default shell
ENV SHELL=/bin/bash

# Create a working directory
WORKDIR /app/

# Add the deadsnakes PPA for Python 3.10
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    python3.10-venv \
    apt-utils \
    wget \
    git

# Update alternatives to set python3 to point to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# alias python='python3'
RUN ln -s /usr/bin/python3.10 /usr/bin/python

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /app/

RUN pip install -r requirements.txt

RUN pip install ninja


CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
EXPOSE 8888
EXPOSE 6006