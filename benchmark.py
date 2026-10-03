#!/usr/bin/env python3
"""Benchmark the v2 streaming pipeline on a WAV file.

Local mode (default): loads the real model once and simulates the daemon's
chunked pseudo-streaming in-process -- chunked VAD-gated partials plus one
final transcribe -- reporting compute cost and latencies.

Socket mode (``--socket``): streams the file through the live daemon paced in
real time, measuring end-to-end first-partial and final latency including IPC.

Example::

    python benchmark.py /tmp/oc-voice-last.wav
    python benchmark.py --socket /tmp/oc-voice-last.wav

    Model: small.en
    Device: cuda
    Audio: 3.67 sec
    First partial: 0.72 sec
    Final (after audio end): 1.08 sec
    RTF: 0.257
"""

from __future__ import annotations

import argparse
import socket
import sys
import time

import numpy as np

from config import load as load_config
from config import resolve_socket


def ensure_cuda_libs() -> None:
    """Re-exec with the venv's NVIDIA runtime on LD_LIBRARY_PATH.

    ctranslate2 loads libcublas at inference time and only honors
    LD_LIBRARY_PATH set *before* the process starts (setting it from inside
    Python is too late). voice-daemon.sh does this for the daemon; the
    benchmark does it for itself so ``python benchmark.py`` just works.
    """
    import glob
    import os

    if os.environ.get("OC_VOICE_CUDA_READY") == "1":
        return
    here = os.path.dirname(os.path.abspath(__file__))
    libs = sorted(glob.glob(os.path.join(
        here, ".venv", "lib", "python*", "site-packages", "nvidia", "*", "lib")))
    if not libs:
        return
    path = ":".join(libs)
    if os.environ.get("LD_LIBRARY_PATH"):
        path += ":" + os.environ["LD_LIBRARY_PATH"]
    os.environ["LD_LIBRARY_PATH"] = path
    os.environ["OC_VOICE_CUDA_READY"] = "1"
    os.execvpe(sys.executable,
               [sys.executable, os.path.abspath(__file__)] + sys.argv[1:],
               os.environ)


def load_wav_mono16(path: str, sr: int) -> np.ndarray:
    from faster_whisper.audio import decode_audio

    return decode_audio(path, sampling_rate=sr).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="benchmark streaming pipeline")
    ap.add_argument("wav", nargs="?", default="/tmp/oc-voice-last.wav")
    ap.add_argument("--config", default=None)
    ap.add_argument("--socket", action="store_true",
                    help="stream through the live daemon instead of in-process")
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--cancel-test", action="store_true",
                    help="measure CANCEL -> CANCELLED reaction through the "
                         "live daemon (needs --socket)")
    ap.add_argument("--compare", action="store_true",
                    help="run FINAL_ONLY and LOW_COST_PARTIAL back to back "
                         "on the same audio (in-process)")
    args = ap.parse_args(argv)
    ensure_cuda_libs()

    if args.cancel_test and not args.socket:
        print("--cancel-test needs --socket")
        return 2

    cfg = load_config(args.config)
    sr = int(cfg["audio"]["sample_rate"])
    audio = load_wav_mono16(args.wav, sr)
    audio_s = len(audio) / sr
    print(f"Audio: {audio_s:.2f} sec ({args.wav})")

    if args.socket:
        if args.cancel_test:
            return run_cancel_test(cfg, audio, args.chunk_ms)
        return run_socket(cfg, args.wav, audio, audio_s, args.chunk_ms)
    if args.compare:
        return run_compare(cfg, audio, audio_s, args.chunk_ms)
    return run_inprocess(cfg, audio, audio_s, args.chunk_ms)


def vram_used() -> str:
    """Best-effort VRAM snapshot (nvidia-smi), else 'n/a'."""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return "n/a"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return f"{out.stdout.strip()} MiB" if out.returncode == 0 else "n/a"
    except Exception:
        return "n/a"


def run_compare(cfg: dict, audio: np.ndarray, audio_s: float,
                chunk_ms: int) -> int:
    """FINAL_ONLY vs LOW_COST_PARTIAL on identical audio, one table."""
    import copy

    print(f"VRAM before: {vram_used()}")
    off = copy.deepcopy(cfg)
    off["streaming"]["enabled"] = False
    print("\n== A: FINAL_ONLY ==")
    ra = run_inprocess(off, audio, audio_s, chunk_ms, quiet=True)
    on = copy.deepcopy(cfg)
    on["streaming"]["enabled"] = True
    on["streaming"]["interval_ms"] = 3000
    on["streaming"]["window_s"] = 4.0
    on["streaming"]["min_new_speech_s"] = 2.0
    print("\n== B: LOW_COST_PARTIAL ==")
    rb = run_inprocess(on, audio, audio_s, chunk_ms, quiet=True)
    print("\n== compare ==")
    print(f"VRAM after: {vram_used()}")
    for label, r in (("FINAL_ONLY", ra), ("PARTIAL", rb)):
        print(f"{label}: compute={r['compute_s']:.2f}s "
              f"asr_calls={r['asr_calls']} partials={r['partials']} "
              f"final={r['final_s']:.2f}s conf={r['final_conf']} "
              f"text={r['text']!r}")
    print("Partials cost extra inference; they never change the final.")
    return 0


def run_inprocess(cfg: dict, audio: np.ndarray, audio_s: float,
                  chunk_ms: int, quiet: bool = False) -> int | dict:
    import importlib.util
    import os
    import sys

    spec = importlib.util.spec_from_file_location(
        "voice_daemon",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "voice-daemon.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["voice_daemon"] = mod
    spec.loader.exec_module(mod)

    from vad_gate import analyze_array, speech_spans

    from asr import Transcriber

    t0 = time.monotonic()
    model, device = mod.load_model(cfg)
    load_s = time.monotonic() - t0
    print(f"Model: {cfg['transcription']['model']}\nDevice: {device} "
          f"(load {load_s:.1f}s, excluded from RTF)")

    tr = Transcriber(model, cfg)
    st = cfg["streaming"]
    interval = float(st["interval_ms"]) / 1000.0
    window_s = float(st["window_s"])
    min_new = float(st["min_new_speech_s"])
    vad_min = float(cfg["vad"]["min_speech_s"])
    trim_save = float(cfg["vad"]["trim_savings_s"])

    chunk = int(sr_len(cfg) * chunk_ms / 1000)
    buffered = 0
    last_partial_end = 0
    first_partial = None
    n_partials = 0
    confs: list = []
    compute_s = 0.0
    asr_calls = 0
    total = len(audio)
    do_partials = bool(cfg["streaming"].get("enabled", False))

    # Simulate paced arrival: process chunk-by-chunk, attempt a partial every
    # interval of (simulated) audio time -- but measure only compute time.
    next_partial_at = interval
    while buffered < total:
        buffered = min(total, buffered + chunk)
        sim_t = buffered / sr_len(cfg)
        if do_partials and sim_t >= next_partial_at:
            next_partial_at += interval
            if buffered - last_partial_end < int(min_new * sr_len(cfg)):
                continue
            tail = audio[max(0, buffered - int(window_s * sr_len(cfg))):buffered]
            t1 = time.monotonic()
            v = analyze_array(tail, sr_len(cfg), tr.vad_options, vad_min)
            vad_cost = time.monotonic() - t1
            compute_s += vad_cost
            if not v.has_speech:
                continue
            t1 = time.monotonic()
            text, conf, _segs = tr.transcribe_confident(tail)
            asr_calls += 1
            compute_s += time.monotonic() - t1
            last_partial_end = buffered
            if text and first_partial is None:
                first_partial = compute_s
                if not quiet:
                    print(f'Partial #{n_partials + 1}: "{text}" (conf={conf})')
            if text:
                n_partials += 1
                confs.append(conf)

    t1 = time.monotonic()
    v = analyze_array(audio, sr_len(cfg), tr.vad_options, vad_min)
    compute_s += time.monotonic() - t1
    final_conf = None
    if v.has_speech:
        if audio_s - v.speech_s >= trim_save:
            spans = speech_spans(audio, sr_len(cfg), tr.vad_options)
            audio = np.concatenate([audio[s["start"]:s["end"]] for s in spans])
        t1 = time.monotonic()
        text, final_conf, _segs = tr.transcribe_confident(audio)
        asr_calls += 1
        final_s = time.monotonic() - t1
    else:
        text, final_s = "", 0.0
    compute_s += final_s

    stats = {"compute_s": compute_s, "asr_calls": asr_calls,
             "partials": n_partials, "final_s": final_s,
             "final_conf": final_conf, "text": text}
    if quiet:
        return stats
    print(f'Transcript: "{text}"')
    report(audio_s, compute_s, first_partial, final_s, n_partials,
           confs, final_conf)
    return 0


def sr_len(cfg: dict) -> int:
    return int(cfg["audio"]["sample_rate"])


def report(audio_s: float, compute_s: float, first_partial,
           final_s: float, n_partials: int, confs: list | None = None,
           final_conf=None) -> None:
    print(f"Partials: {n_partials}")
    print(f"First partial: "
          f"{f'{first_partial:.2f} sec' if first_partial is not None else 'none'}")
    print(f"Final (after audio end): {final_s:.2f} sec")
    print(f"Compute total: {compute_s:.2f} sec")
    print(f"RTF: {compute_s / audio_s:.3f}" if audio_s else "RTF: n/a")
    vals = [c for c in (confs or []) if c is not None]
    if vals:
        print(f"Avg partial confidence: {sum(vals) / len(vals):.3f}")
    print(f"Final confidence: {final_conf}")


def run_socket(cfg: dict, wav: str, audio: np.ndarray, audio_s: float,
               chunk_ms: int) -> int:
    import proto

    sock_path = resolve_socket(cfg)
    sr = int(cfg["audio"]["sample_rate"])
    raw = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    step = sr * chunk_ms // 1000 * 2  # bytes per chunk

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(sock_path)
    except OSError as e:
        print(f"cannot reach daemon at {sock_path}: {e}")
        return 1
    s.settimeout(30)
    rid = proto.new_request_id()
    proto.send_json(s, {"type": "start", "request_id": rid,
                        "sample_rate": sr})
    m = proto.read_json(s)
    if m.get("type") != "started":
        print(f"start refused: {m}")
        return 1
    t0 = time.monotonic()
    first_partial = None
    partials = 0
    pconfs: list = []
    final = None
    final_conf = None
    final_at = None
    pos = 0
    # Catch-up paced send: all audio due by the clock goes out each beat, so
    # measured latencies reflect the server, not sender drift.
    import select as _select

    s.setblocking(False)
    stop_sent = False
    while True:
        now = time.monotonic()
        if not stop_sent:
            due = min(len(raw), int((now - t0) * sr * 2))
            while pos < due:
                proto.send_audio(s, raw[pos:pos + step])
                pos += step
            if pos >= len(raw):
                s.setblocking(True)
                s.settimeout(30)
                proto.send_json(s, {"type": "stop", "request_id": rid})
                stop_sent = True
                t_end = time.monotonic()
        if not stop_sent:
            r, _, _ = _select.select([s], [], [], 0.05)
            if not r:
                continue
            s.setblocking(True)
            s.settimeout(0.05)
            try:
                kind, payload = proto.read_frame(s)
            except socket.timeout:
                s.setblocking(False)
                continue
            s.setblocking(False)
        else:
            s.settimeout(30)
            kind, payload = proto.read_frame(s)
        if kind != proto.FRAME_JSON:
            continue
        try:
            m = proto.loads(payload)
        except proto.ProtocolError:
            continue
        if m.get("type") == "partial":
            partials += 1
            if m.get("confidence") is not None:
                pconfs.append(m["confidence"])
            if first_partial is None:
                first_partial = time.monotonic() - t0
            print(f'Partial #{partials}: "{m.get("text", "")}" '
                  f'(conf={m.get("confidence")})')
        elif m["type"] == "result":
            final = m.get("text", "")
            final_conf = m.get("confidence")
            final_at = time.monotonic()
            timings = m.get("timings", {})
            print(f"Server timings: {timings}")
            break
        else:
            print(f"server said: {m}")
            return 1
    s.close()
    print(f'Transcript: "{final}"')
    print(f"Partials: {partials}")
    print(f"First partial: "
          f"{f'{first_partial:.2f} sec' if first_partial is not None else 'none'}")
    print(f"Final (after audio end): {final_at - t_end:.2f} sec")
    print(f"Wall total: {final_at - t0:.2f} sec (audio {audio_s:.2f}s)")
    if pconfs:
        print(f"Avg partial confidence: {sum(pconfs) / len(pconfs):.3f}")
    print(f"Final confidence: {final_conf} "
          f"({'LOW' if m.get('low_confidence') else 'ok'})")
    return 0


def run_cancel_test(cfg: dict, audio: np.ndarray, chunk_ms: int) -> int:
    """Measure CANCEL -> CANCELLED reaction against the live daemon.

    Streams ~1s of audio, sends CANCEL mid-recording (no STOP), and times the
    response. Target: <0.3s even if the daemon is busy transcribing -- the
    reader answers without waiting for the worker.
    """
    import proto

    sock_path = resolve_socket(cfg)
    sr = int(cfg["audio"]["sample_rate"])
    raw = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    step = sr * chunk_ms // 1000 * 2

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(sock_path)
    except OSError as e:
        print(f"cannot reach daemon at {sock_path}: {e}")
        return 1
    s.settimeout(15)
    rid = proto.new_request_id()
    proto.send_json(s, {"type": "start", "request_id": rid,
                        "sample_rate": sr})
    m = proto.read_json(s)
    if m.get("type") != "started":
        print(f"start refused: {m}")
        return 1
    sent = 0
    target = min(len(raw), sr * 2)  # ~1s of audio
    while sent < target:
        proto.send_audio(s, raw[sent:sent + step])
        sent += step
    t0 = time.monotonic()
    proto.send_json(s, {"type": "cancel", "request_id": rid})
    m = proto.read_json(s)
    dt = time.monotonic() - t0
    print(f"CANCEL -> {m.get('type')} in {dt * 1000:.0f} ms "
          f"(target < 300 ms)")
    s.close()
    if m.get("type") != "cancelled" or m.get("request_id") != rid:
        print("FAIL: unexpected cancel response")
        return 1
    print("PASS" if dt < 0.3 else "SLOW (see above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
