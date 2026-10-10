# Hugging Face Space (Docker SDK). Builds the demo straight from your public GitHub repository,
# so the Space repo only needs this Dockerfile and the README.md next to it.
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

ARG REPO_URL=https://github.com/tarubee19/autoinsight.git
ARG REPO_REF=main
RUN git clone --depth 1 --branch ${REPO_REF} ${REPO_URL} /app

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860 API_URL=http://127.0.0.1:8000 \
    AUTOINSIGHT_LLM_PROVIDER=offline AUTOINSIGHT_SANDBOX_MODE=local AUTOINSIGHT_RUNS_DIR=/tmp/runs \
    AUTOINSIGHT_SAMPLES_DIR=/app/data/sample AUTOINSIGHT_MAX_UPLOAD_MB=10 AUTOINSIGHT_RETENTION_HOURS=24
RUN pip install --no-cache-dir -c constraints.txt ".[api,ui]" && chmod +x docker/start_demo.sh

# Spaces run the container as uid 1000
RUN useradd --create-home --uid 1000 user && mkdir -p /tmp/runs && chown -R user:user /tmp/runs /app
USER user
EXPOSE 7860
CMD ["docker/start_demo.sh"]
