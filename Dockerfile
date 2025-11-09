# ...existing code...
FROM python:3.12-slim

WORKDIR /app

# Dependências mínimas úteis (build-essential para compilar wheels se necessário)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

# use shell form para expansão de ${PORT}; a referência do módulo foi alterada para maingcs:app
CMD ["sh", "-c", "uvicorn maingcs:app --host 0.0.0.0 --port ${PORT:-8080}"]