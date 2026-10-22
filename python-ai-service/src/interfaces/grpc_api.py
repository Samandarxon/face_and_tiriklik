"""gRPC interfeys adapteri — `face.v1.FaceService` implementatsiyasi.

Bu qatlam faqat protokol bilan shug'ullanadi: proto <-> domain o'girish va
xatolarni uzatish. Butun biznes mantiq `application/use_cases.py` da qoladi — HTTP va gRPC
adapterlari AYNAN bir xil use-case'larni chaqiradi.

Xatolarni uzatish:
  * Unary chaqiruvlar — gRPC status + `FaceError` detali (google.rpc.Status).
    Shu tufayli mijoz `spoof_detected` bilan birga liveness ballarini ham oladi.
  * Streaming — xatolar IN-BAND (`FrameResponse.error`). Bitta soxta kadr
    butun oqimni uzib qo'ymasligi kerak.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import AsyncIterator, Optional

import grpc
from google.protobuf import any_pb2
from google.rpc import code_pb2, status_pb2
from grpc_status import rpc_status

from face.v1 import face_pb2, face_pb2_grpc

from ..container import container
from ..domain.entities import DetectedFace, LivenessResult
from ..domain.errors import (
    AIServiceError,
    InvalidImageError,
    LivenessUnavailableError,
    LowQualityFaceError,
    ModelNotReadyError,
    MultipleFacesError,
    NoFaceDetectedError,
    SpoofDetectedError,
)
from ..infrastructure.config import settings
from ..runner import run_inference, run_io

logger = logging.getLogger("ai-service.grpc")

# Domain xatosi -> proto enum + gRPC status kodi.
_ERROR_MAP: dict[type, tuple[int, int]] = {
    InvalidImageError: (face_pb2.ERROR_CODE_INVALID_IMAGE, code_pb2.INVALID_ARGUMENT),
    NoFaceDetectedError: (face_pb2.ERROR_CODE_NO_FACE_DETECTED, code_pb2.FAILED_PRECONDITION),
    MultipleFacesError: (face_pb2.ERROR_CODE_MULTIPLE_FACES, code_pb2.FAILED_PRECONDITION),
    LowQualityFaceError: (face_pb2.ERROR_CODE_LOW_QUALITY_FACE, code_pb2.FAILED_PRECONDITION),
    SpoofDetectedError: (face_pb2.ERROR_CODE_SPOOF_DETECTED, code_pb2.PERMISSION_DENIED),
    LivenessUnavailableError: (face_pb2.ERROR_CODE_LIVENESS_UNAVAILABLE, code_pb2.UNAVAILABLE),
    ModelNotReadyError: (face_pb2.ERROR_CODE_MODEL_NOT_READY, code_pb2.UNAVAILABLE),
}


# ------------------------------------------------------------------ o'girish

def _liveness_proto(result: Optional[LivenessResult]) -> Optional[face_pb2.Liveness]:
    if result is None:
        return None
    return face_pb2.Liveness(
        is_real=result.is_real,
        real_probability=result.real_probability,
        spoof_probability=result.spoof_probability,
        threshold=result.threshold,
        per_model=result.per_model,
    )


def _liveness_from_dict(data: dict) -> face_pb2.Liveness:
    """SpoofDetectedError.details ichidagi liveness dict'idan proto yasaydi."""
    return face_pb2.Liveness(
        is_real=bool(data.get("is_real", False)),
        real_probability=float(data.get("real_probability", 0.0)),
        spoof_probability=float(data.get("spoof_probability", 0.0)),
        threshold=float(data.get("threshold", 0.0)),
        per_model={k: float(v) for k, v in (data.get("per_model") or {}).items()},
    )


def _face_proto(face: DetectedFace) -> face_pb2.FaceInfo:
    x1, y1, x2, y2 = face.bbox.as_tuple()
    info = face_pb2.FaceInfo(
        bbox=face_pb2.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        det_score=face.det_score,
    )
    # Bosh burilishi aktiv liveness uchun. Yo'q bo'lsa available=False bilan
    # yuboriladi — maydonni butunlay tashlab ketish "0 gradus" degan
    # noto'g'ri xulosaga olib kelardi.
    if face.pose is not None:
        info.pose.CopyFrom(face_pb2.HeadPose(
            yaw=face.pose.yaw,
            pitch=face.pose.pitch,
            roll=face.pose.roll,
            available=face.pose.available,
        ))
    return info


def _face_error(exc: AIServiceError) -> face_pb2.FaceError:
    proto_code, _ = _ERROR_MAP.get(type(exc), (face_pb2.ERROR_CODE_INTERNAL, code_pb2.INTERNAL))
    error = face_pb2.FaceError(code=proto_code, message=exc.message)

    for key, value in (exc.details or {}).items():
        if key == "liveness" and isinstance(value, dict):
            error.liveness.CopyFrom(_liveness_from_dict(value))
        else:
            error.details[key] = str(value)
    return error


async def _abort(context: grpc.aio.ServicerContext, exc: AIServiceError) -> None:
    """Unary chaqiruvni `FaceError` detali bilan to'xtatadi."""
    _, grpc_code = _ERROR_MAP.get(type(exc), (None, code_pb2.INTERNAL))

    detail = any_pb2.Any()
    detail.Pack(_face_error(exc))
    rich_status = status_pb2.Status(code=grpc_code, message=exc.message, details=[detail])
    await context.abort_with_status(rpc_status.to_status(rich_status))


# ------------------------------------------------------------------- servicer

class FaceServicer(face_pb2_grpc.FaceServiceServicer):
    """Barcha metodlar `engine` ni chaqiradi; og'ir ish threadpool'da bajariladi."""

    # ------------------------------------------------------------ unary

    async def Embed(self, request: face_pb2.ImageRequest,
                    context: grpc.aio.ServicerContext) -> face_pb2.EmbedResponse:
        request_id = request.request_id or uuid.uuid4().hex[:12]
        try:
            frame = await run_io(container.decoder.decode, request.image)
            face = await run_inference(container.embed_face.execute, frame)
        except AIServiceError as exc:
            logger.info("[%s] Embed -> %s: %s", request_id, exc.code, exc.message)
            await _abort(context, exc)
            return face_pb2.EmbedResponse()  # pragma: no cover (abort chiqib ketadi)

        return face_pb2.EmbedResponse(
            embedding=face.embedding,
            embedding_dim=face.embedding_dim,
            face=_face_proto(face),
        )

    async def VerifyFrame(self, request: face_pb2.ImageRequest,
                          context: grpc.aio.ServicerContext) -> face_pb2.VerifyFrameResponse:
        request_id = request.request_id or uuid.uuid4().hex[:12]
        try:
            frame = await run_io(container.decoder.decode, request.image)
            # STEP 1 — soxta bo'lsa shu yerda SpoofDetectedError ko'tariladi
            # va natija (jumladan embedding) UMUMAN qaytarilmaydi.
            analysis = await run_inference(container.verify_frame.execute, frame)
            face, liveness = analysis.face, analysis.liveness
        except AIServiceError as exc:
            level = logging.WARNING if isinstance(exc, SpoofDetectedError) else logging.INFO
            logger.log(level, "[%s] VerifyFrame -> %s: %s", request_id, exc.code, exc.message)
            await _abort(context, exc)
            return face_pb2.VerifyFrameResponse()  # pragma: no cover

        # STEP 2 — embedding allaqachon detect bosqichida hisoblangan.
        return face_pb2.VerifyFrameResponse(
            embedding=face.embedding,
            embedding_dim=face.embedding_dim,
            face=_face_proto(face),
            liveness=_liveness_proto(liveness),
        )

    async def CheckLiveness(self, request: face_pb2.ImageRequest,
                            context: grpc.aio.ServicerContext) -> face_pb2.LivenessResponse:
        request_id = request.request_id or uuid.uuid4().hex[:12]
        try:
            frame = await run_io(container.decoder.decode, request.image)
            analysis = await run_inference(container.probe_liveness.execute, frame)
            face, liveness = analysis.face, analysis.liveness
        except AIServiceError as exc:
            logger.info("[%s] CheckLiveness -> %s", request_id, exc.code)
            await _abort(context, exc)
            return face_pb2.LivenessResponse()  # pragma: no cover

        return face_pb2.LivenessResponse(face=_face_proto(face), liveness=_liveness_proto(liveness))

    async def Health(self, request: face_pb2.HealthRequest,
                     context: grpc.aio.ServicerContext) -> face_pb2.HealthResponse:
        status = container.status()
        return face_pb2.HealthResponse(
            ready=status.ready,
            device=status.device,
            insightface_model=status.insightface_model,
            embedding_dim=status.embedding_dim,
            liveness_enabled=status.liveness_enabled,
            liveness_loaded=status.liveness_loaded,
            liveness_models=list(status.liveness_models),
            liveness_threshold=status.liveness_threshold,
            liveness_require_unanimous=status.liveness_require_unanimous,
            liveness_error=status.liveness_error or "",
        )

    # -------------------------------------------------------- streaming

    async def StreamFrames(self, request_iterator: AsyncIterator[face_pb2.FrameRequest],
                           context: grpc.aio.ServicerContext) -> AsyncIterator[face_pb2.FrameResponse]:
        """Ikki tomonlama oqim.

        Har bir kadr mustaqil qayta ishlanadi va natija darhol qaytariladi.
        Xato (soxta yuz, yuz yo'q, ...) oqimni UZMAYDI — u `FrameResponse.error`
        ichida keladi va mijoz keyingi kadrni yuboraveradi.
        """
        peer = context.peer()
        logger.info("stream ochildi: %s", peer)
        frames = 0

        try:
            async for request in request_iterator:
                frames += 1
                started = time.perf_counter()
                response = await self._handle_frame(request)
                response.processing_ms = int((time.perf_counter() - started) * 1000)
                yield response
        except Exception:  # pragma: no cover - mijoz uzilishi odatiy holat
            logger.info("stream uzildi: %s (%d kadr)", peer, frames)
            raise
        finally:
            logger.info("stream yopildi: %s (%d kadr)", peer, frames)

    async def _handle_frame(self, request: face_pb2.FrameRequest) -> face_pb2.FrameResponse:
        response = face_pb2.FrameResponse(frame_id=request.frame_id)
        full = request.mode == face_pb2.STREAM_MODE_FULL

        try:
            frame = await run_io(container.decoder.decode, request.image)
            analysis = await run_inference(container.verify_frame.execute, frame)
            face, liveness = analysis.face, analysis.liveness
        except AIServiceError as exc:
            response.ok = False
            response.error.CopyFrom(_face_error(exc))
            # Soxta kadrda ham bbox foydali bo'lardi, lekin uni bilish uchun
            # detect bosqichidan o'tgan bo'lish kerak — spoof holatida o'tgan.
            return response

        response.ok = True
        response.face.CopyFrom(_face_proto(face))
        response.liveness.CopyFrom(_liveness_proto(liveness))

        if full:
            response.embedding.extend(face.embedding)
            response.embedding_dim = face.embedding_dim
        return response


# --------------------------------------------------------------------- server

def _server_credentials() -> Optional[grpc.ServerCredentials]:
    """GRPC_TLS_* dan server sertifikatini yasaydi; sozlanmagan bo'lsa None.

    Faqat bittasi (sertifikat yoki kalit) berilgan bo'lsa — bu xato, jimgina
    ochiq portga tushib qolinmaydi.
    """
    cert, key = settings.grpc_tls_cert_file, settings.grpc_tls_key_file
    if not cert and not key:
        return None
    if not cert or not key:
        raise RuntimeError("GRPC_TLS_CERT_FILE va GRPC_TLS_KEY_FILE birga berilishi kerak")

    with open(key, "rb") as f:
        private_key = f.read()
    with open(cert, "rb") as f:
        certificate_chain = f.read()

    client_ca = None
    if settings.grpc_tls_client_ca_file:
        with open(settings.grpc_tls_client_ca_file, "rb") as f:
            client_ca = f.read()

    logger.info("gRPC TLS yoqilgan (mTLS: %s)", "ha" if client_ca else "yo'q")
    return grpc.ssl_server_credentials(
        [(private_key, certificate_chain)],
        root_certificates=client_ca,
        require_client_auth=client_ca is not None,
    )


async def serve(host: str, port: int) -> grpc.aio.Server:
    """gRPC serverni ishga tushiradi va uni qaytaradi (to'xtatish chaqiruvchida)."""
    server = grpc.aio.server(
        options=[
            # Kadrlar 1-2 MB bo'lishi mumkin; standart 4 MB chegara yetarli,
            # lekin aniq ko'rsatib qo'yamiz.
            ("grpc.max_receive_message_length", settings.max_upload_bytes + (1 << 20)),
            ("grpc.max_send_message_length", 8 << 20),
            # Uzoq turadigan streaming ulanishlar uzilib qolmasligi uchun.
            ("grpc.keepalive_time_ms", 30_000),
            ("grpc.keepalive_timeout_ms", 10_000),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.http2.min_ping_interval_without_data_ms", 10_000),
        ]
    )
    face_pb2_grpc.add_FaceServiceServicer_to_server(FaceServicer(), server)

    # Reflection — `grpcurl` va Postman proto fayli bo'lmasa ham ishlashi uchun.
    try:
        from grpc_reflection.v1alpha import reflection

        reflection.enable_server_reflection(
            (face_pb2.DESCRIPTOR.services_by_name["FaceService"].full_name,
             reflection.SERVICE_NAME),
            server,
        )
    except ImportError:  # pragma: no cover
        logger.warning("grpcio-reflection o'rnatilmagan — reflection o'chirilgan.")

    address = f"{host}:{port}"
    credentials = _server_credentials()
    if credentials is None:
        logger.warning(
            "gRPC TLS'siz tinglamoqda (%s). Faqat yopiq ichki tarmoqda qabul qilinadi — "
            "GRPC_TLS_CERT_FILE/GRPC_TLS_KEY_FILE (va mTLS uchun GRPC_TLS_CLIENT_CA_FILE) qo'ying.",
            address,
        )
        server.add_insecure_port(address)
    else:
        server.add_secure_port(address, credentials)
    await server.start()
    logger.info("gRPC server tinglamoqda: %s", address)
    return server
