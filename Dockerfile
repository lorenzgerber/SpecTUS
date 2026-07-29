# CUDA-enabled base so the same image runs on both CPU-only hosts (Fargate) and
# GPU hosts (EC2) - torch just falls back to CPU when no GPU is visible.
# Pinned to the exact torch/CUDA combo this model was trained and validated with.
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# libxrender1/libxext6: RDKit's Chem.Draw module (pulled in transitively via
# PandasTools, used by spectra_process_utils.py) needs these at import time
# even though this service never renders images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY setup.py .
COPY spectus/ spectus/
COPY tokenizer/ tokenizer/
COPY requirements.txt .

RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir -r requirements.txt

# spectus/ scripts (predict.py, service.py, ...) import each other and model/utils
# as top-level modules (e.g. `from utils.data_utils import ...`), matching how
# they're already run in this repo (`python spectus/predict.py` from repo root).
ENV PYTHONPATH=/app/spectus
WORKDIR /app/spectus

ENV SPECTUS_CHECKPOINT=/app/checkpoint
ENV SPECTUS_TOKENIZER=/app/tokenizer/tokenizer_mf10M.model

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
