FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY producer.py .
COPY consumer.py .

# El comando de ejecución se define en docker-compose.yml