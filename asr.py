#!/usr/bin/env python3
"""ASR service: single persistent model + honest confidence scoring.

Confidence is NOT invented: each faster-whisper segment carries
``avg_logprob`` (mean log-probability per token, e.g. -0.15 is good, -1.5 is
bad) and ``no_speech_prob``. The aggregate is a duration-weighted mean of
``exp(avg_logprob)`` over segments:

    confidence = sum(dur_i * exp(avg_logprob_i)) / sum(dur_i)

``no_speech_prob`` is not multiplied in: VAD already removed silence, and
silence-framed segments would otherwise double-penalize. It IS exposed per
segment for diagnostics. The number is a quality signal, not a calibrated
probability -- treat 0.9 as "model is sure", 0.4 as "guess", nothing more.

Segments missing probability data (mock models, future backends) yield
``confidence=None`` rather than a fabricated number; callers must handle null.
"""

from __future__ import annotations

import math
import threading

import numpy as np


def segment_confidence(seg) -> float | None:
    """Per-segment confidence in [0, 1], or None when data is missing."""
    try:
        lp = float(getattr(seg, "avg_logprob", None))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lp):
        return None
    return min(1.0, max(0.0, math.exp(lp)))


def aggregate_confidence(segments) -> float | None:
    """Duration-weighted mean of per-segment confidence.

    Returns None when no segment carries usable probability data.
    Empty input -> None (callers decide what empty means).
    """
    total, weighted = 0.0, 0.0
    for seg in segments:
        conf = segment_confidence(seg)
        if conf is None:
            continue
        try:
            start = float(getattr(seg, "start", 0.0) or 0.0)
            end = float(getattr(seg, "end", start) or start)
        except (TypeError, ValueError):
            start, end = 0.0, 0.0
        dur = max(0.0, end - start) or 1.0  # untimed segments: equal weight
        total += dur
        weighted += dur * conf
    if total <= 0:
        return None
    return round(weighted / total, 3)


def segment_info(seg) -> dict:
    """Segment-level detail for logs/diagnostics (no audio, just metadata)."""
    info = {
        "start": getattr(seg, "start", None),
        "end": getattr(seg, "end", None),
        "text": getattr(seg, "text", ""),
        "confidence": segment_confidence(seg),
    }
    try:
        info["no_speech_prob"] = round(float(seg.no_speech_prob), 3)
    except (AttributeError, TypeError, ValueError):
        info["no_speech_prob"] = None
    return info


class Transcriber:
    """Single-model transcription with VAD-filtered decoding.

    All model calls are serialized through one lock. The socket server and
    the legacy spool thread SHARE one Transcriber so the GTX 1050 never sees
    concurrent inference from the two paths.
    """

    def __init__(self, model, cfg: dict):
        from vad_gate import options_from_config

        self.model = model
        self.cfg = cfg
        self.lock = threading.Lock()
        self.vad_options = options_from_config(cfg)
        t = cfg["transcription"]
        self.language = t["language"]
        self.beam_size = int(t["beam_size"])
        self.initial_prompt = t["initial_prompt"]
        self.confidence_enabled = bool(t.get("confidence_enabled", True))

    def transcribe(self, audio: np.ndarray):
        """Raw segments (list) for one float32 mono 16kHz array."""
        with self.lock:
            segments, _ = self.model.transcribe(
                np.asarray(audio, dtype=np.float32),
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=True,
                initial_prompt=self.initial_prompt,
            )
            return list(segments)

    def transcribe_array(self, audio: np.ndarray) -> str:
        """Backwards-compatible plain-text transcribe (legacy path)."""
        return " ".join(
            s.text.strip() for s in self.transcribe(audio)).strip()

    def transcribe_confident(self, audio: np.ndarray
                             ) -> tuple[str, float | None, list]:
        """Text + aggregate confidence + raw segments, one model call."""
        segments = self.transcribe(audio)
        text = " ".join(s.text.strip() for s in segments).strip()
        conf = aggregate_confidence(segments) if self.confidence_enabled else None
        return text, conf, segments
