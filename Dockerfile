# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Dependências do sistema (PyMuPDF precisa de mupdf libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
