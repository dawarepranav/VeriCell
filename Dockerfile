# VeriCell web app + agent. Keys come from the environment (never baked into the image):
#   docker compose up      (reads .env)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vericell ./vericell
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1
EXPOSE 8765
CMD ["uvicorn", "vericell.web.app:app", "--host", "0.0.0.0", "--port", "8765"]
