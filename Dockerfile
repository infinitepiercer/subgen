FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# Layer 1: System packages (~200MB, rarely changes)
RUN (apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg python3 python3-pip curl gosu tzdata git || \
    (apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    ffmpeg python3 python3-pip curl gosu tzdata git)) && \
    rm -rf /var/lib/apt/lists/*

# Layer 2: PyTorch (~2.5GB, only changes on torch version bumps)
RUN python3 -m pip install -U --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Layer 3: Python dependencies (~500MB, only changes when requirements.txt changes)
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -U --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

WORKDIR /subgen

# Layer 4: Source code (tiny, changes on every push)
COPY entrypoint.sh /entrypoint.sh
COPY launcher.py subgen.py language_code.py /subgen/
COPY subgen/ /subgen/subgen/

# Cache directory for HuggingFace/Matplotlib
RUN mkdir -p /cache && chmod 777 /cache

ENV XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    MPLCONFIGDIR=/cache/matplotlib \
    WESPEAKER_HOME=/cache/wespeaker \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "launcher.py"]
