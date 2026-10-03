#!/usr/bin/env python3
"""Persistent whisper server: loads the model ONCE, then transcribes queued wavs.

Protocol (spool dir /tmp/oc-voice):
  writer:  req.wav.tmp -> rename req.wav   (atomic handoff)
  daemon:  rename req.wav -> proc.wav, transcribe, res.txt.tmp -> rename res.txt
"""
import os
import sys
import time

from faster_whisper import WhisperModel

MODEL = "small.en"
SPOOL = "/tmp/oc-voice"
INITIAL_PROMPT = (
    "OpenCode on Garuda Linux with Hyprland. "
    "Dictation in the terminal, browser and documents."
)


def main() -> None:
    os.makedirs(SPOOL, exist_ok=True)
    try:
        model = WhisperModel(MODEL, device="cuda", compute_type="int8")
        print("device: cuda", flush=True)
    except Exception as e:
        print(f"cuda failed ({e}), using cpu", flush=True)
        model = WhisperModel(MODEL, device="cpu", compute_type="int8")
    print("ready", flush=True)

    req = os.path.join(SPOOL, "req.wav")
    while True:
        if not os.path.exists(req):
            time.sleep(0.1)
            continue
        job = os.path.join(SPOOL, "proc.wav")
        try:
            os.rename(req, job)
        except OSError:
            continue
        try:
            segments, _ = model.transcribe(
                job,
                language="en",
                beam_size=1,
                vad_filter=True,
                initial_prompt=INITIAL_PROMPT,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:
            print(f"transcribe error: {e}", flush=True)
            text = ""
        try:
            os.remove(job)
        except OSError:
            pass
        tmp = os.path.join(SPOOL, "res.txt.tmp")
        with open(tmp, "w") as f:
            f.write(text)
        os.rename(tmp, os.path.join(SPOOL, "res.txt"))


if __name__ == "__main__":
    main()
