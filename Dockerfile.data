# Dockerfile for BTC RPC Data Container
# This container provides a portable data dump of Bitcoin blockchain data

FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/m-marqx/ML-Miner"
LABEL org.opencontainers.image.description="Bitcoin RPC and OHLCV data container"
LABEL org.opencontainers.image.licenses="MIT"

# Set working directory
WORKDIR /app

# Install minimal dependencies for data access
RUN pip install --no-cache-dir pandas==2.0.3 pyarrow==12.0.1

# Copy data folder
COPY data/ ./data/

# Create a simple Python script to help users explore the data
RUN echo 'import pandas as pd\n\
import os\n\
\n\
def list_data_files():\n\
    """List all parquet files in the data directory."""\n\
    for root, dirs, files in os.walk("data"):\n\
        for file in files:\n\
            if file.endswith(".parquet"):\n\
                print(os.path.join(root, file))\n\
\n\
def load_ohlcv():\n\
    """Load BTC/USDT OHLCV data."""\n\
    return pd.read_parquet("data/cryptodata/dynamic_btc.parquet")\n\
\n\
def load_onchain():\n\
    """Load all on-chain block stats data."""\n\
    return pd.read_parquet("data/onchain/BTC/block_stats_fragments")\n\
\n\
print("BTC RPC Data Container")\n\
print("======================")\n\
print("Available functions:")\n\
print("  list_data_files() - List all parquet files")\n\
print("  load_ohlcv()      - Load BTC/USDT OHLCV data")\n\
print("  load_onchain()    - Load on-chain block stats")\n\
print()\n\
list_data_files()\n\
' > /app/data_explorer.py

# Default command: start Python interactive shell with data explorer
CMD ["python", "-i", "/app/data_explorer.py"]
