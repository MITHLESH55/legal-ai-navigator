FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY req.txt .

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r req.txt && \
    pip install bcrypt

RUN python -m spacy download en_core_web_sm

COPY . .

RUN python -c "from livekit.plugins import silero; silero.VAD.load()" || echo "Silero VAD will load on first use"

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

CMD ["/app/start.sh"]
