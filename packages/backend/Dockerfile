FROM python:3.12-slim
WORKDIR /app
# postgresql-client-16 (matching the pgvector/pgvector:pg16 server image in
# docker-compose.yml, since pg_dump refuses to run against a newer server
# than itself) backs the Reset DB wizard's pre-delete backup
# (app/services/reset_db.py backup_before_reset) -- without this, that
# backup always fails with "command not found" inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg \
    && install -d /usr/share/postgresql-common/pgdg \
    && curl -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc --fail \
        https://www.postgresql.org/media/keys/ACCC4CF8.asc \
    && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(. /etc/os-release && echo $VERSION_CODENAME)-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update && apt-get install -y --no-install-recommends postgresql-client-16 \
    && apt-get purge -y curl gnupg && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
# CPU-only torch: the container has no GPU, and the default Linux wheel pulls in
# multi-GB CUDA packages. Installed first so requirements.txt's torch dependency
# (via sentence-transformers / gliner) is satisfied by the CPU build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
