FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# ASR_ENGINE is a runtime config; both whisper and parakeet deps are always installed.
ARG ASR_ENGINE=whisper

# Layer 1: System packages (~200MB, rarely changes)
ENV DEBIAN_FRONTEND=noninteractive
RUN (apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg python3 python3-pip python3-dev build-essential curl gosu tzdata git libsndfile1 portaudio19-dev || \
    (apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    ffmpeg python3 python3-pip python3-dev build-essential curl gosu tzdata git libsndfile1 portaudio19-dev)) && \
    rm -rf /var/lib/apt/lists/*

# Layer 2: PyTorch (~2.5GB, only changes on torch version bumps)
RUN python3 -m pip install -U --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Layer 3: Python dependencies (~500MB, only changes when requirements.txt changes)
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -U --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Layer 3b: NeMo installation for Parakeet backend
COPY requirements-parakeet.txt /tmp/requirements-parakeet.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements-parakeet.txt && \
    # If NeMo upgraded numpy past v1, rebuild hdbscan without downgrading numpy
    NUMPY_MAJOR=$(python3 -c "import numpy; print(numpy.__version__.split('.')[0])") && \
    if [ "$NUMPY_MAJOR" -ge 2 ]; then \
        python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
            --no-build-isolation --no-binary :all: hdbscan==0.8.37 ; \
    else \
        python3 -m pip install --no-cache-dir --force-reinstall --no-binary hdbscan hdbscan==0.8.37 ; \
    fi && \
    # Ensure torchaudio matches installed torch version
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])") && \
    pip install --no-cache-dir "torchaudio==${TORCH_VER}" \
        --index-url https://download.pytorch.org/whl/cu124 || \
    pip install --no-cache-dir torchaudio && \
    rm /tmp/requirements-parakeet.txt

# Layer 3c: KenLM n-gram tools for Parakeet LM fusion (improves word accuracy)
RUN apt-get update && \
    apt-get install -y --no-install-recommends cmake libboost-program-options-dev \
        libboost-system-dev libboost-thread-dev libboost-test-dev zlib1g-dev libbz2-dev liblzma-dev && \
    rm -rf /var/lib/apt/lists/* && \
    git clone --depth 1 https://github.com/kpu/kenlm.git /tmp/kenlm && \
    cd /tmp/kenlm && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF && make -j$(nproc) && \
    cp bin/lmplz /usr/local/bin/ && \
    rm -rf /tmp/kenlm

# Layer 3d: Flash Attention (GPU-only speedup for ESPnet/transformers attention layers)
RUN pip install --no-cache-dir --prefix=/install \
    flash-attn --no-build-isolation || \
    echo "Flash Attention install failed — continuing without it (fallback to default attention)"

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
    NEMO_CACHE_DIR=/cache/nemo \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
# Layer 5: Install dependencies to build FFmpeg
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    pkg-config \
    yasm \
    nasm \
    libssl-dev \
    zlib1g-dev \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libmp3lame-dev \
    libopus-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and compile FFmpeg 6.0
# --enable-shared is the CRITICAL flag to fix the R_X86_64_PC32 error
RUN get -qO ffmpeg.tar.bz2 https://ffmpeg.org/releases/ffmpeg-6.0.tar.bz2 \
    && tar xjvf ffmpeg.tar.bz2 \
    && cd ffmpeg-6 \
    && ./configure \
    --prefix=/usr/local \
    --enable-shared \
    --disable-static \
    --disable-doc \
    --disable-ffplay \
    --enable-gpl \
    --enable-nonfree \
    --enable-version3 \
    --enable-openssl \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libvpx \
    --enable-libmp3lame \
    --enable-libopus \
    && make -j$(nproc) \
    && make install

EXPOSE 9000
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "launcher.py"]
