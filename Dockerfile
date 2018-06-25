FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get -y update -qq
RUN apt-get install -y python-opencv

RUN python -m pip install -U pip setuptools wheel keras python-opencv

COPY . /app
WORKDIR /app
ENV PYTHONBUFFERED 1
ENV PYTHONPATH /app

ENTRYPOINT /bin/bash