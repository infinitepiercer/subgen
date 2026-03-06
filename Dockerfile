FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# Build arg: set to "parakeet" to include NeMo for the Parakeet ASR backend.
# Default "whisper" keeps the image lean (no NeMo overhead).
ARG ASR_ENGINE=whisper

# Layer 1: System packages (~200MB, rarely changes)
ENV DEBIAN_FRONTEND=noninteractive
RUN (apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg python3 python3-pip python3-dev build-essential curl gosu tzdata git libsndfile1 || \
    (apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    ffmpeg python3 python3-pip python3-dev build-essential curl gosu tzdata git libsndfile1)) && \
    rm -rf /var/lib/apt/lists/*

# Layer 2: PyTorch (~2.5GB, only changes on torch version bumps)
RUN python3 -m pip install -U --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Layer 3: Python dependencies (~500MB, only changes when requirements.txt changes)
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -U --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Layer 3b: Conditional NeMo installation for Parakeet backend
COPY requirements-parakeet.txt /tmp/requirements-parakeet.txt
RUN if [ "$ASR_ENGINE" = "parakeet" ]; then \
        python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
        python3 -m pip install --no-cache-dir -r /tmp/requirements-parakeet.txt && \
        python3 -m pip install --no-cache-dir --force-reinstall --no-binary hdbscan hdbscan==0.8.37 ; \
    fi && \
    rm /tmp/requirements-parakeet.txt

# Layer 3c: KenLM n-gram tools for Parakeet LM fusion (improves word accuracy)
RUN if [ "$ASR_ENGINE" = "parakeet" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends cmake libboost-program-options-dev \
            libboost-system-dev libboost-thread-dev libboost-test-dev zlib1g-dev libbz2-dev liblzma-dev && \
        rm -rf /var/lib/apt/lists/* && \
        git clone --depth 1 https://github.com/kpu/kenlm.git /tmp/kenlm && \
        cd /tmp/kenlm && mkdir build && cd build && \
        cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF && make -j$(nproc) && \
        cp bin/lmplz /usr/local/bin/ && \
        rm -rf /tmp/kenlm ; \
    fi

WORKDIR /subgen

# Layer 4: Source code (tiny, changes on every push)
COPY entrypoint.sh /entrypoint.sh
COPY launcher.py subgen.py language_code.py /subgen/
COPY subgen/ /subgen/subgen/

# Cache directory for HuggingFace/Matplotlib
RUN mkdir -p /cache && chmod 777 /cache

# Expose ASR_ENGINE at runtime so the application can detect the backend
ENV ASR_ENGINE=${ASR_ENGINE} \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    MPLCONFIGDIR=/cache/matplotlib \
    WESPEAKER_HOME=/cache/wespeaker \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "launcher.py"]
