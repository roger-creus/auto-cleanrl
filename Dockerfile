FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get -y install python3-pip python3-dev xvfb ffmpeg git build-essential python3-opengl && \
    rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/bin/python

# install python dependencies
# Create minimal package stubs so uv can resolve the project deps
RUN mkdir -p cleanrl && touch cleanrl/__init__.py && \
    mkdir -p cleanrl_utils && touch cleanrl_utils/__init__.py && \
    touch README.md
RUN pip install uv --upgrade
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
RUN uv pip install --system ".[atari,envpool]"