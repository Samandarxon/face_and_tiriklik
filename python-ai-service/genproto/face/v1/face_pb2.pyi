from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ERROR_CODE_UNSPECIFIED: _ClassVar[ErrorCode]
    ERROR_CODE_INVALID_IMAGE: _ClassVar[ErrorCode]
    ERROR_CODE_NO_FACE_DETECTED: _ClassVar[ErrorCode]
    ERROR_CODE_MULTIPLE_FACES: _ClassVar[ErrorCode]
    ERROR_CODE_LOW_QUALITY_FACE: _ClassVar[ErrorCode]
    ERROR_CODE_SPOOF_DETECTED: _ClassVar[ErrorCode]
    ERROR_CODE_LIVENESS_UNAVAILABLE: _ClassVar[ErrorCode]
    ERROR_CODE_MODEL_NOT_READY: _ClassVar[ErrorCode]
    ERROR_CODE_INTERNAL: _ClassVar[ErrorCode]

class StreamMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STREAM_MODE_UNSPECIFIED: _ClassVar[StreamMode]
    STREAM_MODE_LIVENESS_ONLY: _ClassVar[StreamMode]
    STREAM_MODE_FULL: _ClassVar[StreamMode]
ERROR_CODE_UNSPECIFIED: ErrorCode
ERROR_CODE_INVALID_IMAGE: ErrorCode
ERROR_CODE_NO_FACE_DETECTED: ErrorCode
ERROR_CODE_MULTIPLE_FACES: ErrorCode
ERROR_CODE_LOW_QUALITY_FACE: ErrorCode
ERROR_CODE_SPOOF_DETECTED: ErrorCode
ERROR_CODE_LIVENESS_UNAVAILABLE: ErrorCode
ERROR_CODE_MODEL_NOT_READY: ErrorCode
ERROR_CODE_INTERNAL: ErrorCode
STREAM_MODE_UNSPECIFIED: StreamMode
STREAM_MODE_LIVENESS_ONLY: StreamMode
STREAM_MODE_FULL: StreamMode

class FaceError(_message.Message):
    __slots__ = ("code", "message", "details", "liveness")
    class DetailsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    LIVENESS_FIELD_NUMBER: _ClassVar[int]
    code: ErrorCode
    message: str
    details: _containers.ScalarMap[str, str]
    liveness: Liveness
    def __init__(self, code: _Optional[_Union[ErrorCode, str]] = ..., message: _Optional[str] = ..., details: _Optional[_Mapping[str, str]] = ..., liveness: _Optional[_Union[Liveness, _Mapping]] = ...) -> None: ...

class BoundingBox(_message.Message):
    __slots__ = ("x1", "y1", "x2", "y2")
    X1_FIELD_NUMBER: _ClassVar[int]
    Y1_FIELD_NUMBER: _ClassVar[int]
    X2_FIELD_NUMBER: _ClassVar[int]
    Y2_FIELD_NUMBER: _ClassVar[int]
    x1: int
    y1: int
    x2: int
    y2: int
    def __init__(self, x1: _Optional[int] = ..., y1: _Optional[int] = ..., x2: _Optional[int] = ..., y2: _Optional[int] = ...) -> None: ...

class Liveness(_message.Message):
    __slots__ = ("is_real", "real_probability", "spoof_probability", "threshold", "per_model")
    class PerModelEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    IS_REAL_FIELD_NUMBER: _ClassVar[int]
    REAL_PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    SPOOF_PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    PER_MODEL_FIELD_NUMBER: _ClassVar[int]
    is_real: bool
    real_probability: float
    spoof_probability: float
    threshold: float
    per_model: _containers.ScalarMap[str, float]
    def __init__(self, is_real: bool = ..., real_probability: _Optional[float] = ..., spoof_probability: _Optional[float] = ..., threshold: _Optional[float] = ..., per_model: _Optional[_Mapping[str, float]] = ...) -> None: ...

class HeadPose(_message.Message):
    __slots__ = ("yaw", "pitch", "roll", "available")
    YAW_FIELD_NUMBER: _ClassVar[int]
    PITCH_FIELD_NUMBER: _ClassVar[int]
    ROLL_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    yaw: float
    pitch: float
    roll: float
    available: bool
    def __init__(self, yaw: _Optional[float] = ..., pitch: _Optional[float] = ..., roll: _Optional[float] = ..., available: bool = ...) -> None: ...

class FaceInfo(_message.Message):
    __slots__ = ("bbox", "det_score", "pose")
    BBOX_FIELD_NUMBER: _ClassVar[int]
    DET_SCORE_FIELD_NUMBER: _ClassVar[int]
    POSE_FIELD_NUMBER: _ClassVar[int]
    bbox: BoundingBox
    det_score: float
    pose: HeadPose
    def __init__(self, bbox: _Optional[_Union[BoundingBox, _Mapping]] = ..., det_score: _Optional[float] = ..., pose: _Optional[_Union[HeadPose, _Mapping]] = ...) -> None: ...

class ImageRequest(_message.Message):
    __slots__ = ("image", "request_id")
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    image: bytes
    request_id: str
    def __init__(self, image: _Optional[bytes] = ..., request_id: _Optional[str] = ...) -> None: ...

class EmbedResponse(_message.Message):
    __slots__ = ("embedding", "embedding_dim", "face")
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_DIM_FIELD_NUMBER: _ClassVar[int]
    FACE_FIELD_NUMBER: _ClassVar[int]
    embedding: _containers.RepeatedScalarFieldContainer[float]
    embedding_dim: int
    face: FaceInfo
    def __init__(self, embedding: _Optional[_Iterable[float]] = ..., embedding_dim: _Optional[int] = ..., face: _Optional[_Union[FaceInfo, _Mapping]] = ...) -> None: ...

class VerifyFrameResponse(_message.Message):
    __slots__ = ("embedding", "embedding_dim", "face", "liveness")
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_DIM_FIELD_NUMBER: _ClassVar[int]
    FACE_FIELD_NUMBER: _ClassVar[int]
    LIVENESS_FIELD_NUMBER: _ClassVar[int]
    embedding: _containers.RepeatedScalarFieldContainer[float]
    embedding_dim: int
    face: FaceInfo
    liveness: Liveness
    def __init__(self, embedding: _Optional[_Iterable[float]] = ..., embedding_dim: _Optional[int] = ..., face: _Optional[_Union[FaceInfo, _Mapping]] = ..., liveness: _Optional[_Union[Liveness, _Mapping]] = ...) -> None: ...

class LivenessResponse(_message.Message):
    __slots__ = ("face", "liveness")
    FACE_FIELD_NUMBER: _ClassVar[int]
    LIVENESS_FIELD_NUMBER: _ClassVar[int]
    face: FaceInfo
    liveness: Liveness
    def __init__(self, face: _Optional[_Union[FaceInfo, _Mapping]] = ..., liveness: _Optional[_Union[Liveness, _Mapping]] = ...) -> None: ...

class FrameRequest(_message.Message):
    __slots__ = ("image", "frame_id", "mode", "request_id")
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    image: bytes
    frame_id: int
    mode: StreamMode
    request_id: str
    def __init__(self, image: _Optional[bytes] = ..., frame_id: _Optional[int] = ..., mode: _Optional[_Union[StreamMode, str]] = ..., request_id: _Optional[str] = ...) -> None: ...

class FrameResponse(_message.Message):
    __slots__ = ("frame_id", "ok", "error", "face", "liveness", "embedding", "embedding_dim", "processing_ms")
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    OK_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    FACE_FIELD_NUMBER: _ClassVar[int]
    LIVENESS_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_DIM_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_MS_FIELD_NUMBER: _ClassVar[int]
    frame_id: int
    ok: bool
    error: FaceError
    face: FaceInfo
    liveness: Liveness
    embedding: _containers.RepeatedScalarFieldContainer[float]
    embedding_dim: int
    processing_ms: int
    def __init__(self, frame_id: _Optional[int] = ..., ok: bool = ..., error: _Optional[_Union[FaceError, _Mapping]] = ..., face: _Optional[_Union[FaceInfo, _Mapping]] = ..., liveness: _Optional[_Union[Liveness, _Mapping]] = ..., embedding: _Optional[_Iterable[float]] = ..., embedding_dim: _Optional[int] = ..., processing_ms: _Optional[int] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("ready", "device", "insightface_model", "embedding_dim", "liveness_enabled", "liveness_loaded", "liveness_models", "liveness_threshold", "liveness_require_unanimous", "liveness_error")
    READY_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    INSIGHTFACE_MODEL_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_DIM_FIELD_NUMBER: _ClassVar[int]
    LIVENESS_ENABLED_FIELD_NUMBER: _ClassVar[int]
    LIVENESS_LOADED_FIELD_NUMBER: _ClassVar[int]
    LIVENESS_MODELS_FIELD_NUMBER: _ClassVar[int]
    LIVENESS_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    LIVENESS_REQUIRE_UNANIMOUS_FIELD_NUMBER: _ClassVar[int]
    LIVENESS_ERROR_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    device: str
    insightface_model: str
    embedding_dim: int
    liveness_enabled: bool
    liveness_loaded: bool
    liveness_models: _containers.RepeatedScalarFieldContainer[str]
    liveness_threshold: float
    liveness_require_unanimous: bool
    liveness_error: str
    def __init__(self, ready: bool = ..., device: _Optional[str] = ..., insightface_model: _Optional[str] = ..., embedding_dim: _Optional[int] = ..., liveness_enabled: bool = ..., liveness_loaded: bool = ..., liveness_models: _Optional[_Iterable[str]] = ..., liveness_threshold: _Optional[float] = ..., liveness_require_unanimous: bool = ..., liveness_error: _Optional[str] = ...) -> None: ...
