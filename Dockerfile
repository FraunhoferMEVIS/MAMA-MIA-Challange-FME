FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

LABEL maintainer="kai.geissler@mevis.fraunhofer.de"

ADD requirements.txt .
RUN pip install -r requirements.txt 