# syntax=docker/dockerfile:1.5
# ==============================

# base cuda image
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04 as base

# install python
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
        libsndfile1 \
      && \
    rm -rf /var/lib/apt/lists/

# setup user
ARG USER=developer

RUN adduser $USER
USER $USER
ENV HOME=/home/$USER
ENV PATH=$HOME/.local/bin:$PATH
WORKDIR $HOME/app

# install python requirements
COPY requirements.txt .
RUN --mount=type=cache,target=$HOME/.cache \
  pip install --user --upgrade pip wheel setuptools && \
  pip install --user -r requirements.txt

# default command
CMD [ "bash" ]

# ==============================

# TODO: install development tools for dev image
# TODO: install cuda
# base, base-devel, base-prod
# base-cuda, base-cuda-devel, base-cuda-prod
