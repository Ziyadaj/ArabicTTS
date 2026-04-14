# CUDA 12.1 base — matches the torch==2.2.2+cu121 wheels pinned in pyproject.
# Works on Ada (RTX 4080, sm_89).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TTS_HOME=/workspace/.cache/tts \
    COQUI_TOS_AGREED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        git curl ca-certificates ffmpeg libsndfile1 espeak-ng \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN pip install --no-cache-dir uv

WORKDIR /workspace

# Copy only dependency metadata first so image layers cache on code edits.
COPY pyproject.toml README.md ./
COPY src ./src

# Install the heavy GPU stack. Torch wheels come from the PyTorch cu121 index.
RUN uv pip install --system --extra-index-url https://download.pytorch.org/whl/cu121 \
        -e '.[tts,dev]'

COPY scripts ./scripts
COPY tests ./tests

EXPOSE 7860

CMD ["arabic-tts", "serve", "--host", "0.0.0.0", "--port", "7860"]
