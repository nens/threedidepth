FROM ubuntu:noble
LABEL maintainer="arjan.verkerk@nelen-schuurmans.nl"

RUN apt-get update && apt-get install --yes \
    git \
    locales \
    python3-pip \
    python3-full \
    python3-gdal \
    python3-dev \
    libcairo2-dev \
    pkg-config \
    libgirepository1.0-dev

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8

RUN userdel -r ubuntu
ARG uid=1000
ARG gid=1000
RUN groupadd -g $gid nens && useradd -lm -u $uid -g $gid nens

VOLUME /code
WORKDIR /code
USER nens
ENV PATH=/code/.venv/bin:$PATH
