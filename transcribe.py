#!/usr/bin/env python3
"""Transcribe a 16kHz mono wav with local faster-whisper small.en (GPU if usable)."""
import glob
import os
import sys
import sysconfig

# CUDA runtime ships as pip wheels (nvidia-*); expose them before ctranslate2 loads.
_purelib = sysconfig.get_paths()["purelib"]
_nvlibs = sorted(glob.glob(os.path.join(_purelib, "nvidia", "*", "lib")))
if _nvlibs:
    os.environ["LD_LIBRARY_PATH"] = (
        ":".join(_nvlibs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )

from faster_whisper import WhisperModel


MODEL = "small.en"
# Vocabulary bias for this setup: tech terms the user dictates often.
INITIAL_PROMPT = (
    "OpenCode on Garuda Linux with Hyprland. "
    "Dictation in the terminal, browser and documents."
)


def load_model() -> tuple[WhisperModel, str]:
    try:
        model = WhisperModel(MODEL, device="cuda", compute_type="int8")
        print("device: cuda", file=sys.stderr)
        return model, "cuda"
    except Exception as e:
        print(f"device: cpu (cuda load failed: {e})", file=sys.stderr)
        return WhisperModel(MODEL, device="cpu", compute_type="int8"), "cpu"


def transcribe_wav(model: WhisperModel, wav: str) -> str:
    segments, _ = model.transcribe(
        wav,
        language="en",
        beam_size=1,
        vad_filter=True,
        initial_prompt=INITIAL_PROMPT,
    )
    return " ".join(s.text.strip() for s in segments).strip()


def main() -> None:
    wav = sys.argv[1] if len(sys.argv) > 1 else "/tmp/oc-voice.wav"
    # Pipeline VAD gate first: silence exits fast WITHOUT loading the model
    # (saves the ~5s reload that used to happen on every empty utterance).
    if os.environ.get("OC_VOICE_NO_VAD_GATE") != "1":
        try:
            from vad_gate import TRIM_SAVINGS_S, analyze, write_speech_only

            vad = analyze(wav)
            if not vad.has_speech:
                print(f"vad: silence ({vad.total_s:.2f}s), skipped whisper",
                      file=sys.stderr)
                print("")
                return
            if vad.total_s - vad.speech_s >= TRIM_SAVINGS_S:
                trimmed = wav + ".speech.wav"
                write_speech_only(wav, trimmed)
                wav = trimmed
                print(f"vad: speech {vad.speech_s:.2f}s/{vad.total_s:.2f}s, "
                      f"trimmed", file=sys.stderr)
        except SystemExit:
            raise
        except Exception as e:
            print(f"vad gate failed ({e}), transcribing full clip",
                  file=sys.stderr)
    model, device = load_model()
    try:
        text = transcribe_wav(model, wav)
    except Exception as e:
        # e.g. CUDA lib missing at inference time: retry once on CPU
        print(f"cuda inference failed, retrying on cpu: {e}", file=sys.stderr)
        model = WhisperModel(MODEL, device="cpu", compute_type="int8")
        text = transcribe_wav(model, wav)
    print(text)


if __name__ == "__main__":
    main()
