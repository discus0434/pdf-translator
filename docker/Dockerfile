FROM paddlepaddle/paddle:2.4.1-gpu-cuda11.2-cudnn8.2-trt8.0

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS=yes
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y \
        poppler-utils \
        libpoppler-dev \
        wget git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/* \
    && mkdir -p /home/pdf-translator

WORKDIR /home/pdf-translator

RUN pip install --no-cache-dir opencv-python-headless \
    networkx \
    transformers \
    sentencepiece \
    paddleocr \
    pdf2image \
    fastapi[uvicorn] \
    PyPDF2 \
    matplotlib \
    && pip install --no-cache-dir torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    torchaudio==0.12.1+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN wget https://github.com/adobe-fonts/source-han-serif/raw/release/OTF/Japanese/SourceHanSerif-Light.otf -P /home/
