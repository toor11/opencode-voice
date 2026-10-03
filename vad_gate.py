#!/usr/bin/env python3
"""Pipeline VAD gate: Silero VAD *before* Whisper, not just inside it.

Whisper's ``vad_filter=True`` only skips silence during decoding -- the GPU
still wakes up, decodes the whole file, and can hallucinate on pure silence
(several empty-transcript GPU runs are visible in /tmp/oc-voice.log).

This module runs the same Silero model (bundled ``silero_vad_v6.onnx``,
CPU/onnxruntime, no extra dependency) up front:

    Microphone wav -> VAD gate -> speech? -> Whisper (or fast empty reply)

It answers two questions:

1. ``analyze()`` -- is there any speech at all? If not, callers skip Whisper
   entirely: no GPU wake, no 5s reload, no hallucination.
2. ``write_speech_only()`` -- concatenate the speech spans into a shorter
   wav so Whisper never sees the leading/trailing silence either.

Tuning differs from faster-whisper's decoding defaults on purpose: the
defaults use ``min_silence_duration_ms=2000`` (glues everything into one
chunk, good for decoding) while the gate uses 500ms (splits pauses, good
for detecting speech vs. tap/breath) plus ``min_speech_duration_ms=250`` to
throw out key clicks and mic pops.

CLI (for shell polling, e.g. voice-auto.sh)::

    vad_gate.py file.wav [--write speech.wav] [--json]
    exit 0 = speech, 2 = silence/no audio, 1 = error
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

# Pipeline VAD preset: sensitive to short commands, drops sub-word blips,
# splits on pauses >= 0.5s. (faster-whisper's decode default merges across
# 2s gaps -- right for decoding, wrong for gating/endpointing.)
GATE_VAD = VadOptions(
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=200,
)

# Below this much speech the clip counts as silence (tap, breath, click).
MIN_SPEECH_S = 0.25

# Rewriting the wav is only worth it when it saves at least this much audio.
TRIM_SAVINGS_S = 0.5

SR = 16000


@dataclass
class VadResult:
    has_speech: bool
    speech_s: float
    total_s: float
    segments: list


def options_from_config(cfg: dict) -> VadOptions:
    """Build pipeline VAD options from a loaded config dict."""
    v = cfg["vad"]
    return VadOptions(
        threshold=float(v["threshold"]),
        min_speech_duration_ms=int(v["min_speech_duration_ms"]),
        min_silence_duration_ms=int(v["min_silence_duration_ms"]),
        speech_pad_ms=int(v["speech_pad_ms"]),
    )


def analyze_array(audio, sample_rate: int = SR,
                  vad_options: VadOptions | None = None,
                  min_speech_s: float = MIN_SPEECH_S) -> VadResult:
    """Run pipeline VAD on an in-memory float32 mono array (socket path)."""
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    total_s = len(audio) / sample_rate
    segments = get_speech_timestamps(audio, vad_options=vad_options or GATE_VAD)
    speech_s = sum((s["end"] - s["start"]) / sample_rate for s in segments)
    has_speech = speech_s >= min_speech_s
    segs = [
        {"start": round(s["start"] / sample_rate, 2),
         "end": round(s["end"] / sample_rate, 2)}
        for s in segments
    ]
    return VadResult(has_speech, round(speech_s, 2), round(total_s, 2), segs)


def speech_spans(audio, sample_rate: int = SR,
                 vad_options: VadOptions | None = None):
    """Raw speech segments (sample offsets) for trimming/endpointing."""
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    return get_speech_timestamps(audio, vad_options=vad_options or GATE_VAD)


def analyze(path: str) -> VadResult:
    """Run pipeline VAD on a wav file. Raises on unreadable audio."""
    audio = decode_audio(path, sampling_rate=SR)
    return analyze_array(audio, SR)


def write_speech_only(src: str, dst: str) -> float:
    """Write concatenated speech spans of src to dst. Returns speech seconds."""
    import numpy as np
    import soundfile as sf

    audio = decode_audio(src, sampling_rate=SR)
    segments = get_speech_timestamps(audio, vad_options=GATE_VAD)
    if not segments:
        raise ValueError("no speech to write")
    speech = np.concatenate([audio[s["start"] : s["end"]] for s in segments])
    sf.write(dst, speech, SR, subtype="PCM_16")
    return round(len(speech) / SR, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav", help="16kHz mono wav (any av-decodable audio works)")
    ap.add_argument("--write", metavar="OUT", default=None,
                    help="also write speech-only audio to OUT")
    ap.add_argument("--json", action="store_true",
                    help="print machine-readable result")
    args = ap.parse_args()

    try:
        res = analyze(args.wav)
    except Exception as e:
        print(f"vad_gate: error reading {args.wav}: {e}", file=sys.stderr)
        return 1

    if args.write and res.has_speech:
        try:
            write_speech_only(args.wav, args.write)
        except Exception as e:
            print(f"vad_gate: trim failed: {e}", file=sys.stderr)

    if args.json:
        print(json.dumps({
            "has_speech": res.has_speech,
            "speech_s": res.speech_s,
            "total_s": res.total_s,
            "segments": res.segments,
        }))
    else:
        verdict = "speech" if res.has_speech else "silence"
        print(f"{verdict}: {res.speech_s:.2f}s speech / {res.total_s:.2f}s audio "
              f"({len(res.segments)} spans)")
    return 0 if res.has_speech else 2


if __name__ == "__main__":
    sys.exit(main())
