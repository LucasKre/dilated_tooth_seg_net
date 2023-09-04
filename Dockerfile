FROM nvidia/cuda:11.6.0-base-ubuntu20.04

ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set bash as the default shell
ENV SHELL=/bin/bash

# Create a working directory
WORKDIR /app/

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    apt-utils \
    vim \
    git

# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get install -y python3-opengl

RUN pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

RUN pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu116_pyt1130/download.html

RUN pip install torch-geometric

RUN apt-get install -y libboost-dev
RUN apt-get install -y python3-opengl

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt


RUN mkdir workdir

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
EXPOSE 8888 6007
