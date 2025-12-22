# ---- Base image ----
FROM python:3.11-slim

# Prevent Python from writing pyc files + force stdout/stderr unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- System deps (for sentence-transformers + basic build tooling) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Poetry ----
ENV POETRY_VERSION=2.1.4
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# ---- Workdir ----
WORKDIR /app

# Copy dependency files first (better Docker layer caching)
COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi --no-root

# Copy the rest of the project
COPY . /app

# Expose API port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "llm_engineering.application.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
