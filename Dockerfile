FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y libsm6 libxext6 libxrender-dev python3.8 python3-pip openssh-server nano rsync

COPY . /app/
WORKDIR /app

RUN cd /app
RUN pip3 install -r requirements_cu116.txt
RUN python3 setup.py install

ENV DGLBACKEND=pytorch
ENV PYTHONPATH /app
