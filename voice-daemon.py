#!/usr/bin/env python3
"""Persistent voice server: loads the Whisper model ONCE, serves the Unix
socket and (optionally) the legacy WAV spool.

Normal IPC is a Unix domain socket (default
``$XDG_RUNTIME_DIR/opencode-voice.sock``, fallback ``/tmp/oc-voice/voice.sock``;
see ``config.py``). One recording session == one connection::

    client: {"type":"start", "request_id":...}   (JSON frame)
    server: {"type":"started", "request_id":...}
    client: <raw s16le mono 16kHz audio frames>  (binary frames)
    server: {"type":"partial", "request_id":..., "text":...}   (replaces prev)
    client: {"type":"stop" | "cancel", "request_id":...}
    server: {"type":"result"|"cancelled"|"error", ...}

No polling: the accept loop and session loop block on the socket. A second
concurrent session gets ``{"type":"busy"}`` and is disconnected -- one active
transcription is all this box (GTX 1050 2GB) wants.

Streaming is chunked pseudo-streaming (see README): partials transcribe a
bounded recent window every ``streaming.interval_ms``; the final result is one
full transcribe of the VAD-trimmed buffer. faster-whisper's ``transcribe()``
is file/array-at-once -- it cannot emit tokens incrementally, so this is the
honest shape of "streaming" on this stack.

Legacy file protocol (req.wav -> proc.wav -> res.txt polling) is kept behind
``--legacy-file-mode`` / ``OC_VOICE_LEGACY=1`` / config
``[server] legacy_file_mode`` for fallback only.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import threading
import time

import numpy as np

import proto
from config import load as load_config
from config import resolve_socket
from normalizer import normalize, rules_from_config
from vad_gate import analyze, write_speech_only

MODEL = "small.en"
SPOOL = "/tmp/oc-voice"

# Partials only fire while speech ended recently (seconds before window end).
# Older speech was already reported; re-decoding it during a pause produces
# prompt-biased echoes, not information.
RECENT_SPEECH_S = 1.5


def load_model(cfg: dict):
    """Load Whisper once. Returns (model, device_name). Never reload after."""
    from faster_whisper import WhisperModel

    want = cfg["transcription"]["device"]
    ctype = cfg["transcription"]["compute_type"]
    model_name = cfg["transcription"]["model"]
    if want in ("auto", "cuda"):
        try:
            model = WhisperModel(model_name, device="cuda", compute_type=ctype)
            print("device: cuda", flush=True)
            return model, "cuda"
        except Exception as e:
            if want == "cuda":
                raise
            print(f"cuda failed ({e}), using cpu", flush=True)
    else:
        print("device: cpu (configured)", flush=True)
    return WhisperModel(model_name, device="cpu", compute_type=ctype), "cpu"


from asr import Transcriber


class VoiceServer:
    def __init__(self, cfg: dict, model, socket_path: str | None = None):
        self.cfg = cfg
        if isinstance(model, Transcriber):
            self.transcriber = model  # shared (e.g. with the legacy thread)
        else:
            self.transcriber = Transcriber(model, cfg)
        self.socket_path = socket_path or resolve_socket(cfg)
        self._listen: socket.socket | None = None
        self._active_lock = threading.Lock()
        self._active_id: str | None = None
        self._shutdown = threading.Event()
        self.device = "?"

    # -- socket setup ----------------------------------------------------
    def _bind(self):
        path = self.socket_path
        parent = os.path.dirname(path)
        os.makedirs(parent, exist_ok=True)
        # Stale socket from a crashed daemon: unlink only if nobody listens.
        if os.path.exists(path):
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(path)
            except OSError:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            else:
                probe.close()
                raise RuntimeError(f"socket already in use: {path}")
            finally:
                probe.close()
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
        srv.listen(8)
        srv.settimeout(0.5)  # wake periodically to notice shutdown
        self._listen = srv
        return srv

    def serve_forever(self) -> None:
        srv = self._bind()
        print(f"listening on {self.socket_path}", flush=True)
        print("ready", flush=True)
        while not self._shutdown.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,),
                             daemon=True).start()
        self._cleanup()

    def shutdown(self) -> None:
        self._shutdown.set()
        try:
            if self._listen is not None:
                self._listen.close()
        except OSError:
            pass
        finally:
            self._listen = None

    def _cleanup(self) -> None:
        try:
            if self.socket_path and os.path.exists(self.socket_path):
                # Only unlink our own socket: connect-test would succeed if
                # another live server owned it (we never get here then).
                os.unlink(self.socket_path)
        except OSError:
            pass

    # -- sessions ----------------------------------------------------------
    def _busy_claim(self, rid: str) -> bool:
        with self._active_lock:
            if self._active_id is not None:
                return False
            self._active_id = rid
            return True

    def _busy_release(self, rid: str) -> None:
        with self._active_lock:
            if self._active_id == rid:
                self._active_id = None

    def _handle(self, conn: socket.socket) -> None:
        session = Session(self, conn)
        session.run()


class Session:
    """One recording session: socket reader thread + ASR worker thread.

    The reader NEVER blocks on the model: it only moves bytes and control
    messages. All blocking ASR runs on the worker. CANCEL is therefore
    answered by the reader in milliseconds even mid-transcribe; the worker
    sees the cancel event and suppresses any late result (the connection is
    closed anyway, and request state dies with the session, so request B can
    never receive request A's text).
    """

    def __init__(self, server: VoiceServer, conn: socket.socket):
        self.server = server
        self.conn = conn
        self.cfg = server.cfg
        self.tr = server.transcriber
        self.rid = "?"
        self.sr = int(self.cfg["audio"]["sample_rate"])
        st = self.cfg["streaming"]
        self.streaming = bool(st.get("enabled", True))
        self.interval = float(st["interval_ms"]) / 1000.0
        self.window_s = float(st["window_s"])
        self.min_new = float(st["min_new_speech_s"])
        self.vad_on = bool(self.cfg["vad"]["enabled"])
        self.vad_min_speech = float(self.cfg["vad"]["min_speech_s"])
        self.trim_save = float(self.cfg["vad"]["trim_savings_s"])
        self.low_thresh = float(
            self.cfg["transcription"].get("low_confidence_threshold", 0.55))
        self.norm_rules = rules_from_config(self.cfg)
        self.cancel_ok = bool(self.cfg.get("cancellation", {}).get(
            "enabled", True))

        self.chunks: list = []
        self.buffered = 0
        self.buf_lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.cancel_ev = threading.Event()
        self.stop_ev = threading.Event()
        self.done_ev = threading.Event()
        self.t_start = time.monotonic()
        self.t_stop: float | None = None
        self.last_partial_text = ""
        self.last_partial_end = 0
        self.n_partials = 0
        self.t_first_partial: float | None = None

    # -- helpers ---------------------------------------------------------
    def log(self, msg: str) -> None:
        print(f"[voice] request={self.rid} {msg}", flush=True)

    def send(self, obj: dict) -> None:
        with self.send_lock:
            proto.send_json(self.conn, obj)

    def snapshot(self) -> np.ndarray:
        with self.buf_lock:
            if not self.chunks:
                return np.zeros(0, np.float32)
            return np.concatenate(self.chunks)

    def finalize_text(self, text: str) -> tuple[str, int]:
        return normalize(text, self.norm_rules)

    def low(self, conf: float | None) -> bool:
        return conf is not None and conf < self.low_thresh

    # -- entry -----------------------------------------------------------
    def run(self) -> None:
        try:
            self.conn.settimeout(15.0)
            try:
                msg = proto.read_json(self.conn)
            except proto.ConnClosed:
                return
            if msg.get("type") != "start" or "request_id" not in msg:
                try:
                    self.send({"type": "error", "request_id": "?",
                               "message": "first message must be start"})
                except OSError:
                    pass
                return
            self.rid = str(msg["request_id"])
            if msg.get("sample_rate", self.sr) != self.sr:
                try:
                    self.send({"type": "error", "request_id": self.rid,
                               "message": f"sample_rate must be {self.sr}"})
                except OSError:
                    pass
                return
            if not self.server._busy_claim(self.rid):
                try:
                    self.send({"type": "busy",
                               "message": "Another transcription is active"})
                except OSError:
                    pass
                return
            try:
                self.t_start = time.monotonic()
                self.send({"type": "started", "request_id": self.rid})
                self.log("started")
                worker = threading.Thread(target=self._worker, daemon=True)
                worker.start()
                self._reader()
                # Free the busy slot before reaping the worker: an abandoned
                # transcribe (cancelled mid-call) must not block the next
                # press. The worker's sends are suppressed; the model lock
                # serializes any overlap. Release is idempotent.
                self.server._busy_release(self.rid)
                worker.join(timeout=120)
            finally:
                self.server._busy_release(self.rid)
        except proto.ProtocolError as e:
            self.log(f"protocol error: {e}")
            try:
                self.send({"type": "error", "request_id": self.rid,
                           "message": str(e)})
            except OSError:
                pass
        except Exception as e:
            self.log(f"handler error: {e}")
            try:
                self.send({"type": "error", "request_id": self.rid,
                           "message": "transcription failed"})
            except OSError:
                pass
        finally:
            self.cancel_ev.set()
            self.stop_ev.set()
            try:
                self.conn.close()
            except OSError:
                pass

    # -- reader: bytes + control only, never blocks on ASR ----------------
    def _reader(self) -> None:
        self.conn.settimeout(self.interval)
        while not self.done_ev.is_set():
            try:
                kind, payload = proto.read_frame(self.conn)
            except socket.timeout:
                continue
            except proto.ConnClosed:
                self.log("client disconnected")
                self.cancel_ev.set()
                self.stop_ev.set()
                return
            if kind == proto.FRAME_AUDIO:
                if payload and not self.stop_ev.is_set():
                    with self.buf_lock:
                        self.chunks.append(
                            proto.s16_bytes_to_float32(payload))
                        self.buffered += len(self.chunks[-1])
                continue
            try:
                cmsg = proto.loads(payload)
            except proto.ProtocolError:
                try:
                    self.send({"type": "error", "request_id": self.rid,
                               "message": "malformed control message"})
                except OSError:
                    pass
                self.cancel_ev.set()
                self.stop_ev.set()
                return
            if not isinstance(cmsg, dict) or \
                    cmsg.get("request_id", self.rid) != self.rid:
                try:
                    self.send({"type": "error", "request_id": self.rid,
                               "message": "request_id mismatch"})
                except OSError:
                    pass
                self.cancel_ev.set()
                self.stop_ev.set()
                return
            ctype = cmsg.get("type")
            if ctype == "stop":
                if self.stop_ev.is_set():
                    continue  # duplicate stop; worker already finalizing
                self.t_stop = time.monotonic()
                audio_s = self.buffered / self.sr
                self.log(f"stop ({audio_s:.2f}s audio)")
                self.stop_ev.set()
            elif ctype == "cancel":
                self._do_cancel()
                return
            else:
                try:
                    self.send({"type": "error", "request_id": self.rid,
                               "message": f"unknown message type: {ctype}"})
                except OSError:
                    pass
                self.cancel_ev.set()
                self.stop_ev.set()
                return

    def _do_cancel(self) -> None:
        """Answer CANCEL immediately; the worker abandons its ASR output."""
        if not self.cancel_ok:
            # Cancellation disabled: treat as STOP (still finalize).
            self.t_stop = time.monotonic()
            self.stop_ev.set()
            return
        self.cancel_ev.set()
        self.stop_ev.set()
        try:
            self.send({"type": "cancelled", "request_id": self.rid})
        except OSError:
            pass
        self.log("cancelled")
        self.done_ev.set()

    # -- worker: the only thread that touches the model -------------------
    def _worker(self) -> None:
        from vad_gate import analyze_array, speech_spans

        win_samples = int(self.window_s * self.sr)
        while not self.stop_ev.is_set():
            if self.cancel_ev.is_set():
                return
            if self.stop_ev.wait(self.interval):
                break
            if self.cancel_ev.is_set() or not self.streaming:
                continue
            with self.buf_lock:
                new = self.buffered - self.last_partial_end
                tail = (np.concatenate(self.chunks)[-win_samples:]
                        if self.chunks else None)
            if new < int(self.min_new * self.sr) or tail is None \
                    or not len(tail):
                continue
            try:
                if self.vad_on:
                    v = analyze_array(tail, self.sr, self.tr.vad_options,
                                      self.vad_min_speech)
                    if not v.has_speech:
                        continue
                    tail_end = len(tail) / self.sr
                    if not any(tail_end - s["end"] <= RECENT_SPEECH_S
                               for s in v.segments):
                        continue
                text, conf, _segs = self.tr.transcribe_confident(tail)
            except Exception as e:
                self.log(f"partial failed: {e}")
                continue
            with self.buf_lock:
                self.last_partial_end = self.buffered
            if self.cancel_ev.is_set():
                return
            if text and text != self.last_partial_text:
                text, _n = self.finalize_text(text)
                if text == self.last_partial_text:
                    continue
                self.last_partial_text = text
                self.n_partials += 1
                if self.t_first_partial is None:
                    self.t_first_partial = time.monotonic()
                    fp = self.t_first_partial - self.t_start
                    self.log(f"first_partial={fp:.2f}s confidence={conf}")
                try:
                    self.send({"type": "partial", "request_id": self.rid,
                               "text": text, "confidence": conf})
                except OSError:
                    return
        if self.cancel_ev.is_set():
            return
        self._finalize()

    def _finalize(self) -> None:
        from vad_gate import analyze_array, speech_spans

        assert self.t_stop is not None
        try:
            audio = self.snapshot()
            duration_s = len(audio) / self.sr
            t0 = time.monotonic()
            text, conf, segments = "", None, []
            if self.vad_on and len(audio):
                v = analyze_array(audio, self.sr, self.tr.vad_options,
                                  self.vad_min_speech)
                vad_s = time.monotonic() - t0
                if not v.has_speech:
                    vad_only = True
                    self.log(f"silence ({duration_s:.2f}s), skipped whisper")
                else:
                    vad_only = False
                    if duration_s - v.speech_s >= self.trim_save:
                        spans = speech_spans(audio, self.sr,
                                             self.tr.vad_options)
                        audio = np.concatenate(
                            [audio[s["start"]:s["end"]] for s in spans])
                        self.log(f"speech {v.speech_s:.2f}s/"
                                 f"{duration_s:.2f}s trimmed")
            elif not len(audio):
                vad_s, vad_only = 0.0, True
                self.log("empty recording")
            else:
                vad_s, vad_only = 0.0, False
            asr_s = 0.0
            if not vad_only:
                if self.cancel_ev.is_set():
                    return
                t1 = time.monotonic()
                text, conf, segments = self.tr.transcribe_confident(audio)
                asr_s = time.monotonic() - t1
            if self.cancel_ev.is_set():
                return  # cancelled mid-ASR: never emit a late result
            text, n_norm = self.finalize_text(text)
            t_done = time.monotonic()
            timings = {
                "duration_s": round(duration_s, 2),
                "first_partial_s": (
                    round(self.t_first_partial - self.t_start, 2)
                    if self.t_first_partial else None),
                "partials": self.n_partials,
                "vad_s": round(vad_s, 2),
                "asr_s": round(asr_s, 2),
                "finalization_s": round(t_done - self.t_stop, 2),
                "total_s": round(t_done - self.t_start, 2),
                "confidence": conf,
            }
            fp = (f" first_partial={timings['first_partial_s']}s"
                  if timings["first_partial_s"] is not None else "")
            self.log(f"final confidence={conf} norm={n_norm}{fp} "
                     f"final_latency={timings['finalization_s']:.2f}s "
                     f"total={timings['total_s']:.2f}s chars={len(text)}")
            try:
                self.send({"type": "result", "request_id": self.rid,
                           "text": text, "confidence": conf,
                           "low_confidence": self.low(conf),
                           "timings": timings})
            except OSError:
                pass
        finally:
            self.done_ev.set()

def legacy_loop(cfg: dict, transcriber: Transcriber,
                stop: threading.Event) -> None:
    """Old WAV spool protocol, fallback only. Polls; socket path never does."""
    spool = cfg["server"]["spool_dir"]
    os.makedirs(spool, exist_ok=True)
    trim_save = float(cfg["vad"]["trim_savings_s"])
    req = os.path.join(spool, "req.wav")
    print("legacy file mode: watching", spool, flush=True)
    while not stop.is_set():
        if not os.path.exists(req):
            stop.wait(0.1)
            continue
        job = os.path.join(spool, "proc.wav")
        try:
            os.rename(req, job)
        except OSError:
            continue
        try:
            vad = analyze(job)
        except Exception as e:
            print(f"vad error ({e}), sending full clip to whisper", flush=True)
            vad = None
        if vad is not None and not vad.has_speech:
            print(f"vad: silence ({vad.total_s:.2f}s), skipped whisper",
                  flush=True)
            try:
                os.remove(job)
            except OSError:
                pass
            tmp = os.path.join(spool, "res.txt.tmp")
            with open(tmp, "w") as f:
                f.write("")
            os.rename(tmp, os.path.join(spool, "res.txt"))
            continue
        if vad is not None and vad.total_s - vad.speech_s >= trim_save:
            try:
                trimmed = job + ".speech.wav"
                write_speech_only(job, trimmed)
                os.rename(trimmed, job)
                print(f"vad: speech {vad.speech_s:.2f}s/{vad.total_s:.2f}s, "
                      f"trimmed", flush=True)
            except Exception as e:
                print(f"vad trim failed ({e}), using full clip", flush=True)
        try:
            text = transcriber.transcribe_array(
                _decode_file(job, transcriber))
        except Exception as e:
            print(f"transcribe error: {e}", flush=True)
            text = ""
        try:
            os.remove(job)
        except OSError:
            pass
        tmp = os.path.join(spool, "res.txt.tmp")
        with open(tmp, "w") as f:
            f.write(text)
        os.rename(tmp, os.path.join(spool, "res.txt"))


def _decode_file(path: str, transcriber: Transcriber):
    from faster_whisper.audio import decode_audio

    return decode_audio(path, sampling_rate=transcriber.cfg["audio"]["sample_rate"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="opencode-voice persistent server")
    ap.add_argument("--config", default=None)
    ap.add_argument("--legacy-file-mode", action="store_true",
                    help="also serve the old req.wav/res.txt spool protocol")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.legacy_file_mode:
        cfg["server"]["legacy_file_mode"] = True

    model, device = load_model(cfg)
    transcriber = Transcriber(model, cfg)
    server = VoiceServer(cfg, transcriber)
    server.device = device

    stop = threading.Event()

    def _sig(signum, frame):
        print(f"signal {signum}, shutting down", flush=True)
        stop.set()
        server.shutdown()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    legacy = None
    if cfg["server"]["legacy_file_mode"]:
        legacy = threading.Thread(target=legacy_loop,
                                  args=(cfg, transcriber, stop), daemon=True)
        legacy.start()
    try:
        server.serve_forever()
    finally:
        stop.set()
        if legacy:
            legacy.join(timeout=2)


if __name__ == "__main__":
    main()
