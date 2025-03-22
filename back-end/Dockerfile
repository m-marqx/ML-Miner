FROM python:3.11-slim

WORKDIR /

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY btc_updater.py .
COPY custom_exceptions/ ./custom_exceptions/
COPY utils/ ./utils/
COPY data/cryptodata ./data/cryptodata
COPY data/onchain ./data/onchain

RUN mkdir -p data/cryptodata data/onchain/BTC/block_stats_fragments/incremental

CMD ["python", "btc_updater.py"]