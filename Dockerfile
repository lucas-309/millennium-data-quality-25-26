FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backtester ./backtester
COPY strategies ./strategies
COPY simulation ./simulation
COPY data_cache ./data_cache
COPY sp500_universe.pkl ./sp500_universe.pkl

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "python -m simulation.backend.server --host ${HOST} --port ${PORT}"]
