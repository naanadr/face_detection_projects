# version 1.0

FROM python:3.6-slim
LABEL maintainer = "fsandrade25@gmail.com"
WORKDIR /app/
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
    build-essential cmake \
    libgtk-3-dev \
    libboost-all-dev \
    && python -m pip install --upgrade pip \
    && pip3 install -r requirements.txt
COPY . /app/
CMD /bin/bash
