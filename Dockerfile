FROM tensorflow/tensorflow:latest-gpu-py3
ENV PYTHONBUFFERED 1

RUN apt-get -y update -qq
RUN apt-get install -y python-opencv python3-dev python3-pip

RUN python3 -m pip install -U pip setuptools wheel
# Тут можно указывать необходимые библиотеки
RUN python3 -m pip install -U keras opencv-python h5py

WORKDIR /project
ENV PYTHONPATH /project

ENTRYPOINT /bin/bash

