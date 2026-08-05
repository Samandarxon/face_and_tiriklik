# CPU build. GPU uchun: Dockerfile.gpu
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    INSIGHTFACE_ROOT=/models/insightface \
    LIVENESS_MODEL_DIR=/app/resources/anti_spoof_models

# insightface kompilyatsiya talab qiladi (Cython), opencv esa libgl/libglib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bog'liqliklar alohida qatlamda — kod o'zgarganda qayta o'rnatilmaydi.
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install "numpy==1.26.4" "cython" \
    && pip install -r requirements.txt

COPY . .

# Anti-spoofing vaznlari image ichida bo'lmasa, build vaqtida yuklab olamiz.
RUN bash scripts/download_models.sh "$LIVENESS_MODEL_DIR"

# buffalo_l (~300 MB) ni build vaqtida yuklab olamiz — birinchi so'rov tez bo'lishi uchun.
RUN python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l', root='/models/insightface', \
                   allowed_modules=['detection','recognition']); \
app.prepare(ctx_id=-1, det_size=(640,640)); \
print('buffalo_l tayyor')"

EXPOSE 8000 50051

HEALTHCHECK --interval=20s --timeout=5s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
