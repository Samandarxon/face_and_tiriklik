"""Python AI Service — composition root.

Bitta jarayonda ikkita protokol ishlaydi:

  gRPC  :50051  — Golang backend AYNAN shu orqali ulanadi
                  (`proto/face/v1/face.proto`). Unary metodlar +
                  `StreamFrames` ikki tomonlama oqimi.
  HTTP  :8000   — Swagger hujjatlari (`/docs`), qo'lda tekshirish, monitoring.

Ikkalasi ham bitta `src/application/use_cases.py` ni chaqiradi — biznes mantiq
takrorlanmaydi, modellar esa xotiraga bir marta yuklanadi.

Qatlamlar (Clean Architecture):
  src/domain/          — ob'ektlar, xatolar, portlar (freymvorksiz)
  src/application/     — use-case'lar (portlarga bog'lanadi)
  src/infrastructure/  — InsightFace, MiniFASNet, OpenCV adapterlari
  src/interfaces/      — gRPC va HTTP adapterlari
  src/container.py     — bog'liqliklarni ulash
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import uvicorn

# Generatsiya qilingan protobuf kodi (`scripts/gen-proto.sh`) import qilinishi uchun.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "genproto"))

from src.container import container  # noqa: E402
from src.infrastructure.config import settings  # noqa: E402
from src.interfaces.http_api import create_app  # noqa: E402
from src.runner import run_io  # noqa: E402

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai-service")


@asynccontextmanager
async def lifespan(_):
    logger.info("Modellar yuklanmoqda...")
    started = time.perf_counter()
    await run_io(container.load)
    logger.info("Modellar %.2f soniyada tayyor bo'ldi. Holat: %s",
                time.perf_counter() - started, container.status().as_dict())

    # gRPC serverni o'sha event loop ichida ko'taramiz.
    grpc_server = None
    if settings.grpc_enabled:
        from src.interfaces.grpc_api import serve as serve_grpc

        grpc_server = await serve_grpc(settings.grpc_host, settings.grpc_port)

    yield

    logger.info("AI service to'xtatilmoqda.")
    if grpc_server is not None:
        # Ochiq streaming ulanishlar yopilishiga imkon beramiz.
        await grpc_server.stop(grace=5)


app = create_app(lifespan)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
