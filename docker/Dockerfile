FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    CUDA_HOME=/usr/local/cuda-11.8 \
    DEBCONF_NOWARNINGS=yes
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y \
        poppler-utils \
        libpoppler-dev \
        wget \
        curl \
        git \
        python3-dev \
        python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

WORKDIR /app

RUN pip install \
        torch \
        torchvision \
        --index-url https://download.pytorch.org/whl/cu118 \
    && pip install opencv-python-headless \
        networkx \
        transformers==4.30 \
        sentencepiece \
        pdf2image \
        fastapi[uvicorn] \
        uvicorn \
        PyPDF2 \
        matplotlib \
        timm \
        scipy \
        shapely \
        pyclipper \
        scikit-image \
        python-multipart \
        onnxruntime \
        requests \
        gradio

RUN pip install \
        "git+https://github.com/facebookresearch/detectron2.git"

RUN git clone https://github.com/microsoft/unilm.git /unilm \
        && sed -i 's/from collections import Iterable/from collections.abc import Iterable/' \
        /unilm/dit/object_detection/ditod/table_evaluation/data_structure.py

RUN wget https://github.com/adobe-fonts/source-han-serif/raw/release/OTF/Japanese/SourceHanSerif-Light.otf -P /app/models \
        && mkdir -p /app/models/unilm \
        && wget "https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-b_cascade.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D" -P /app/models/unilm
