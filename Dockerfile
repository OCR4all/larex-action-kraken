# --- BUILDER ---
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        git \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Application layer
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# --- RUNTIME ---
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && useradd \
        --create-home \
        --uid 10001 \
        --shell /usr/sbin/nologin \
        larex-action \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src /app/src

USER 10001:10001
EXPOSE 9000
CMD ["uvicorn", "larex_action_kraken.main:app", "--host", "0.0.0.0", "--port", "9000"]
