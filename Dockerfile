# BookToAnime — default-stack Docker image.
#
# Includes ffmpeg + tesseract system binaries.
# Default install pulls only permissively-licensed deps.
# To enable local TTS / image generation install the [kokoro] / [visual]
# extras as a layer above this image (see README -> Docker section).
#
# Build:   docker build -t booktoanime .
# Run:     docker run --rm -it -p 8765:8765 \
#            -v "$HOME/booktoanime-data:/data" \
#            -v "$PWD/config.yaml:/config.yaml" \
#            -e BOOKTOANIME_DATA_DIR=/data \
#            -e GROQ_API_KEY="$GROQ_API_KEY" \
#            booktoanime --config /config.yaml run --host 0.0.0.0 --no-open-browser

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    BOOKTOANIME_DATA_DIR=/data

# System binaries used at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better Docker-layer caching.
COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --upgrade pip \
    && pip install /app

# Default data dir is mounted from the host.
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8765

# Drop privileges.
RUN useradd --create-home --shell /usr/sbin/nologin booktoanime \
    && chown -R booktoanime:booktoanime /data /app
USER booktoanime

ENTRYPOINT ["booktoanime"]
CMD ["run", "--host", "0.0.0.0", "--no-open-browser"]
