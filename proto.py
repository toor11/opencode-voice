#!/usr/bin/env python3
"""Unix-socket IPC framing for opencode-voice.

Wire format: every frame is ``kind(1 byte) + length(4 bytes BE) + payload``.

* kind ``0x01``: UTF-8 JSON control message (no newlines needed, length
  delimited so partial reads and message coalescing are handled).
* kind ``0x02``: raw binary PCM audio, signed 16-bit little-endian, mono,
  16 kHz. No base64, no per-chunk JSON envelope.

Control messages (all carry ``request_id`` except ``busy``):

    client -> server: {"type": "start", "request_id": ..., "sample_rate": 16000}
    client -> server: {"type": "stop", "request_id": ...}
    client -> server: {"type": "cancel", "request_id": ...}
    client <- server: {"type": "started", "request_id": ...}
    client <- server: {"type": "partial", "request_id": ..., "text": ...,
                       "confidence": 0.84 | null}
    client <- server: {"type": "result", "request_id": ..., "text": ...,
                       "confidence": 0.91 | null, "low_confidence": false,
                       "timings": {...}}
    client <- server: {"type": "cancelled", "request_id": ...}
    client <- server: {"type": "error", "request_id": ..., "message": ...}
    client <- server: {"type": "busy", "message": ...}  (then server closes)

Semantics: a ``partial`` REPLACES the previous partial; only ``result`` is
authoritative. Audio frames belong to the active request on that connection
(one recording session == one connection), so they carry no request_id.
"""

from __future__ import annotations

import json
import socket
import struct
import uuid
from datetime import datetime

import numpy as np

FRAME_JSON = 0x01
FRAME_AUDIO = 0x02

_HEADER = struct.Struct("!BI")  # kind:1, length:4 (big-endian)
MAX_JSON_BYTES = 1 << 20       # 1 MiB: control messages are tiny
MAX_AUDIO_BYTES = 1 << 24      # 16 MiB: ~8 min of 16kHz mono s16

SAMPLE_RATE = 16000


class ConnClosed(Exception):
    """Peer closed the connection mid-frame."""


class ProtocolError(Exception):
    """Malformed frame (bad kind, absurd length, bad JSON)."""


def new_request_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


def encode_frame(kind: int, payload: bytes) -> bytes:
    return _HEADER.pack(kind, len(payload)) + payload


def send_frame(sock: socket.socket, kind: int, payload: bytes) -> None:
    sock.sendall(encode_frame(kind, payload))


def send_json(sock: socket.socket, obj: dict) -> None:
    send_frame(sock, FRAME_JSON, json.dumps(obj).encode("utf-8"))


def send_audio(sock: socket.socket, pcm_s16: bytes) -> None:
    send_frame(sock, FRAME_AUDIO, pcm_s16)


def _recvn(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes, tolerating partial reads. Raises ConnClosed."""
    chunks = []
    remaining = n
    while remaining:
        try:
            data = sock.recv(remaining)
        except socket.timeout:
            raise
        if not data:
            raise ConnClosed(f"closed after {n - remaining}/{n} bytes")
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def read_frame(sock: socket.socket) -> tuple[int, bytes]:
    """Read one frame. Raises ConnClosed, socket.timeout, ProtocolError."""
    try:
        header = _recvn(sock, _HEADER.size)
    except ConnClosed as e:
        raise e
    kind, length = _HEADER.unpack(header)
    if kind == FRAME_JSON:
        limit = MAX_JSON_BYTES
    elif kind == FRAME_AUDIO:
        limit = MAX_AUDIO_BYTES
    else:
        raise ProtocolError(f"unknown frame kind 0x{kind:02x}")
    if length > limit:
        raise ProtocolError(f"frame too large: kind=0x{kind:02x} len={length}")
    payload = _recvn(sock, length) if length else b""
    return kind, payload


def loads(payload: bytes) -> dict:
    """Decode a JSON control payload. Raises ProtocolError when malformed."""
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"bad JSON control message: {e}") from e
    if not isinstance(obj, dict) or "type" not in obj:
        raise ProtocolError("control message must be a JSON object with 'type'")
    return obj


def read_json(sock: socket.socket) -> dict:
    """Read one frame and decode it as JSON. Audio frames are rejected."""
    kind, payload = read_frame(sock)
    if kind != FRAME_JSON:
        raise ProtocolError("expected JSON control message, got audio frame")
    return loads(payload)


def s16_bytes_to_float32(raw: bytes) -> np.ndarray:
    """Raw s16le mono bytes -> float32 in [-1, 1]. Empty input -> empty array."""
    if not raw:
        return np.array([], dtype=np.float32)
    if len(raw) % 2:
        raw = raw[:-1]  # tolerate a torn trailing byte, don't die on it
    return (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0).copy()
