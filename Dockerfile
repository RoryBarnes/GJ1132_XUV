# Reproducibility Dockerfile for the GJ 1132 XUV Evolution workflow.
# Written as part of the AICS Level 3 envelope; reviewed by the
# researcher before publication. The pinned digest reproduces the
# base image used by the analysis container as of 2026-06-11.
FROM ubuntu:24.04@sha256:c4a8d5503dfb2a3eb8ab5f807da5bc69a85730fb49b5cfca2330194ebcc41c7b

ENV SOURCE_DATE_EPOCH=1780959156
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3=3.12.3-0ubuntu2.1 \
        python3-pip=24.0+dfsg-1ubuntu1.3 \
        git=1:2.43.0-1ubuntu7.3
RUN rm -rf /var/lib/apt/lists/*

# Scientific toolchain: versions are pinned by requirements.lock and
# the binary hashes recorded in .vaibify/environment.json. pytest is
# installed system-wide so test commands resolve in non-login shells
# (the recreated-container PATH drift of 2026-06-10).
COPY requirements.lock /tmp/requirements.lock
RUN python3 -m pip install --break-system-packages --no-cache-dir \
        -r /tmp/requirements.lock \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        pytest==9.0.3

ENV PATH="/home/researcher/.local/bin:${PATH}"
WORKDIR /workspace/GJ1132_XUV
