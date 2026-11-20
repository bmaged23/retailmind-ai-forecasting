FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    UV_PROJECT_ENVIRONMENT=/opt/retailmind-venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
    HF_HOME=/opt/retailmind-model-cache \
    TIMESTRAL_BASE_MODEL_DIR=/opt/retailmind-model-cache/models--mistralai--Timestral-209M/snapshots/current \
    TIMESTRAL_ADAPTER_DIR=/app/models/timestral209m_lora \
    TIMESTRAL_CALIBRATION_FILE=/app/models/timestral209m_lora/calibration.json \
    TIMESTRAL_MODEL_REVISION=9874d252ab0aae16f7f7c999f98f98d6ad7d8365 \
    TIMESTRAL_MAX_GPU_MIB=2048

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY models/timestral209m_lora ./models/timestral209m_lora
COPY apps/dashboard ./apps/dashboard
COPY configs/milestone4_scenarios.json configs/milestone6_ops.json ./configs/
COPY scripts/ops ./scripts/ops

RUN apt-get update \
    && apt-get install --no-install-recommends -y git libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.12.24 \
    && uv sync --locked --no-dev --extra app --extra retailmind-model \
    && mkdir -p /opt/retailmind-model-cache/models--mistralai--Timestral-209M/snapshots/current \
    && HF_HOME=/opt/retailmind-model-cache /opt/retailmind-venv/bin/python -c "from huggingface_hub import snapshot_download; import shutil; shutil.copytree(snapshot_download('mistralai/Timestral-209M', revision='9874d252ab0aae16f7f7c999f98f98d6ad7d8365', cache_dir='/tmp/hf-download'), '/opt/retailmind-model-cache/models--mistralai--Timestral-209M/snapshots/current', dirs_exist_ok=True)" \
    && rm -rf /tmp/uv-cache /tmp/hf-download \
    && useradd --system --create-home --uid 10001 retailmind \
    && mkdir -p /app/artifacts/milestone5 /app/artifacts/milestone6/monitoring \
    && chown -R retailmind:retailmind /app/artifacts /opt/retailmind-venv

RUN chmod 644 /app/models/timestral209m_lora/adapter_model.safetensors

ENV PATH="/opt/retailmind-venv/bin:${PATH}"
USER 10001:10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/ready', timeout=3)"

CMD ["uvicorn", "retailmind.dashboard_api:app", "--host", "0.0.0.0", "--port", "8000"]
