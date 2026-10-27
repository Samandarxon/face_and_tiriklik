import hashlib
import os

import pytest

from src.infrastructure import model_integrity as mi


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_known_file_with_matching_hash_passes(tmp_path, monkeypatch):
    path = _write(tmp_path, "w600k_r50.onnx", b"model")
    monkeypatch.setitem(mi.KNOWN_SHA256, "w600k_r50.onnx", hashlib.sha256(b"model").hexdigest())
    mi.verify_file(path)


def test_tampered_file_is_rejected(tmp_path):
    path = _write(tmp_path, "2.7_80x80_MiniFASNetV2.pth", b"not the real weights")
    with pytest.raises(mi.ModelIntegrityError):
        mi.verify_file(path)


def test_unknown_file_is_rejected(tmp_path):
    path = _write(tmp_path, "injected.onnx", b"x")
    with pytest.raises(mi.ModelIntegrityError):
        mi.verify_file(path)


def test_extra_hash_file_allows_custom_model(tmp_path, monkeypatch):
    path = _write(tmp_path, "custom_int8.onnx", b"q")
    sums = tmp_path / "sums.txt"
    sums.write_text(f"{hashlib.sha256(b'q').hexdigest()}  custom_int8.onnx\n")
    monkeypatch.setenv("MODEL_SHA256_FILE", str(sums))
    mi.verify_file(path)


def test_verify_dir_checks_every_onnx(tmp_path, monkeypatch):
    _write(tmp_path, "w600k_r50.onnx", b"model")
    monkeypatch.setitem(mi.KNOWN_SHA256, "w600k_r50.onnx", hashlib.sha256(b"model").hexdigest())
    _write(tmp_path, "evil.onnx", b"x")
    with pytest.raises(mi.ModelIntegrityError):
        mi.verify_dir(str(tmp_path), ".onnx")


@pytest.mark.skipif(not os.path.isdir("resources/anti_spoof_models"), reason="modellar yo'q")
def test_shipped_liveness_models_match():
    mi.verify_dir("resources/anti_spoof_models", ".pth")
