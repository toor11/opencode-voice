#!/usr/bin/env python3
"""Mic -> Unix-socket streaming client for opencode-voice.

One recording session == one process == one socket connection::

    voice_stream.py [--auto]          # started by voice-start.sh / voice-auto.sh
    # ... speaks ...
    kill -USR1 <pid>                  # voice-stop.sh: stop, wait for result
    # result text lands in $SPOOL/stream.res (+ stream.err on fatal errors)

Signals: SIGUSR1 = STOP (finalize, wait for result); SIGTERM/SIGINT = CANCEL.
``--auto`` endpoints locally with pipeline VAD (trailing silence submits, no
release key needed) and exits on its own; timeouts abort a silent room without
ever bothering the daemon.

Mic backends: ``parec`` (raw s16le stdout, preferred) then ``arecord``.
Partial transcripts from the server replace each other; only the final result
is kept. Exit 0 always carries a result file (possibly empty); exit 1 writes
stream.err and means "don't trust stream.res".
"""

from __future__ import annotations

import argparse
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np

import proto
from config import load as load_config
from config import resolve_socket


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}\n"
    try:
        with open("/tmp/oc-voice.log", "a") as f:
            f.write(line)
    except OSError:
        pass
    print(line.rstrip(), flush=True)


def spawn_mic(sample_rate: int, channels: int) -> subprocess.Popen:
    cmds = [
        ["parec", "--format=s16le", f"--rate={sample_rate}",
         f"--channels={channels}", "--latency-msec=50"],
        ["arecord", "-f", "S16_LE", "-r", str(sample_rate), "-c",
         str(channels), "-t", "raw", "-q"],
    ]
    last_err = ""
    for cmd in cmds:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL)
            # Fail fast if the backend exits immediately (no mic / no server).
            time.sleep(0.2)
            if proc.poll() is not None:
                last_err = f"{cmd[0]} exited at startup"
                continue
            return proc
        except FileNotFoundError:
            last_err = f"{cmd[0]} not installed"
        except OSError as e:
            last_err = f"{cmd[0]}: {e}"
    raise RuntimeError(f"microphone unavailable ({last_err})")


class Streamer:
    def __init__(self, cfg: dict, sock_path: str, auto: bool = False,
                 end_silence: float = 1.2, no_speech_timeout: float = 12.0,
                 max_hold: float = 45.0):
        self.cfg = cfg
        self.sock_path = sock_path
        self.auto = auto
        self.end_silence = end_silence
        self.no_speech_timeout = no_speech_timeout
        self.max_hold = max_hold
        self.sr = int(cfg["audio"]["sample_rate"])
        self.chunk_ms = int(cfg["audio"]["chunk_ms"])
        self.chunk_bytes = self.sr * self.chunk_ms // 1000 * 2  # s16 mono
        spool = cfg["server"]["spool_dir"]
        self.pid_file = os.path.join(spool, "stream.pid")
        self.req_file = os.path.join(spool, "stream.req")
        self.res_file = os.path.join(spool, "stream.res")
        self.conf_file = os.path.join(spool, "stream.conf")
        self.err_file = os.path.join(spool, "stream.err")
        self.rid = proto.new_request_id()
        self.stop_ev = threading.Event()
        self.cancel_requested = False
        self.sock: socket.socket | None = None
        self.inbox: queue.Queue = queue.Queue()
        self.final: dict | None = None
        self.audio_buf = bytearray()
        self.audio_samples = 0

    # -- tiny file helpers -------------------------------------------------
    def _write(self, path: str, text: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.rename(tmp, path)

    def fail(self, message: str) -> int:
        log(f"stream {self.rid}: ERROR {message}")
        try:
            self._write(self.err_file, message)
        except OSError:
            pass
        return 1

    # -- server reader ------------------------------------------------------
    def _reader(self) -> None:
        assert self.sock is not None
        try:
            while True:
                try:
                    msg = proto.read_json(self.sock)
                except proto.ConnClosed:
                    log(f"stream {self.rid}: reader: disconnected")
                    self.inbox.put({"type": "_disconnected"})
                    return
                except (proto.ProtocolError, OSError) as e:
                    log(f"stream {self.rid}: reader: read failed: {e!r}")
                    self.inbox.put({"type": "_error",
                                    "message": f"socket read failed: {e}"})
                    return
                self.inbox.put(msg)
                if msg.get("type") in ("result", "cancelled", "error", "busy"):
                    log(f"stream {self.rid}: reader: terminal {msg.get('type')}")
                    return
        except BaseException as e:  # never die silently: _drain waits on inbox
            log(f"stream {self.rid}: reader: unexpected death: {e!r}")
            self.inbox.put({"type": "_error",
                            "message": f"reader died: {e!r}"})
            raise

    def _drain(self, deadline_s: float = 90.0) -> None:
        """Consume server messages; returns when a terminal one arrives.

        Bounded: a wedged daemon (alive but never answering) fails instead of
        hanging the client -- and voice-stop.sh -- forever.
        """
        end = time.monotonic() + deadline_s
        while True:
            try:
                msg = self.inbox.get(timeout=max(0.1, end - time.monotonic()))
            except queue.Empty:
                if time.monotonic() >= end:
                    self.final = {"type": "_error",
                                  "message": "daemon answer timeout"}
                    return
                continue
            mtype = msg.get("type")
            if mtype == "partial":
                log(f"stream {self.rid}: partial: {msg.get('text', '')}")
            elif mtype in ("result", "cancelled", "error", "busy",
                           "_disconnected", "_error"):
                self.final = msg
                return

    # -- endpointing (auto mode) --------------------------------------------
    def _endpoint_reached(self) -> tuple[bool, str]:
        from faster_whisper.vad import VadOptions

        from vad_gate import speech_spans

        if self.audio_samples < self.sr:  # need >=1s before judging
            return False, ""
        audio = np.frombuffer(bytes(self.audio_buf),
                              dtype=np.int16).astype(np.float32) / 32768.0
        v = self.cfg["vad"]
        opts = VadOptions(
            threshold=float(v["threshold"]),
            min_speech_duration_ms=int(v["min_speech_duration_ms"]),
            min_silence_duration_ms=int(v["min_silence_duration_ms"]),
            speech_pad_ms=int(v["speech_pad_ms"]),
        )
        try:
            spans = speech_spans(audio, self.sr, opts)
        except Exception:
            return False, ""
        total_s = self.audio_samples / self.sr
        speech_s = sum((s["end"] - s["start"]) / self.sr for s in spans)
        if speech_s < float(v["min_speech_s"]):
            if total_s >= self.no_speech_timeout:
                return True, f"no speech in {total_s:.0f}s"
            return False, ""
        trailing = total_s - spans[-1]["end"] / self.sr
        if trailing >= self.end_silence:
            return True, f"silence {trailing:.1f}s after speech"
        return False, ""

    # -- main ---------------------------------------------------------------
    def run(self) -> int:
        for f in (self.res_file, self.conf_file, self.err_file):
            try:
                os.remove(f)
            except OSError:
                pass
        self._write(self.pid_file, str(os.getpid()))
        self._write(self.req_file, self.rid)

        sock = None
        for attempt in range(5):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self.sock_path)
            except OSError as e:
                return self.fail(
                    f"daemon unavailable at {self.sock_path} ({e})")
            self.sock = sock
            try:
                proto.send_json(sock, {"type": "start",
                                       "request_id": self.rid,
                                       "sample_rate": self.sr})
                # Peek at the first reply without consuming partials logic.
                sock.settimeout(10)
                first = proto.read_json(sock)
            except (OSError, proto.ConnClosed, proto.ProtocolError) as e:
                try:
                    sock.close()
                except OSError:
                    pass
                return self.fail(f"start rejected ({e})")
            except socket.timeout:
                try:
                    sock.close()
                except OSError:
                    pass
                return self.fail("daemon did not answer start")
            if first.get("type") == "started":
                self.inbox.put(first)
                break
            # Busy: a previous session is still tearing down; reconnect and
            # retry rather than failing a rapid re-press.
            try:
                sock.close()
            except OSError:
                pass
            if first.get("type") != "busy" or attempt == 4:
                return self.fail(first.get("message", "daemon refused start")
                                 if isinstance(first, dict)
                                 else "daemon refused start")
            log(f"stream {self.rid}: daemon busy, retry {attempt + 1}/5")
            time.sleep(0.3)
        else:
            return self.fail("daemon busy")

        assert sock is not None
        # Stream phase blocks: clear the handshake timeout BEFORE the reader
        # thread starts, otherwise its first recv can inherit the stale 10s
        # timeout and die mid-recording on any utterance longer than that.
        sock.settimeout(None)
        self.sock = sock

        reader = threading.Thread(target=self._reader, daemon=True)
        reader.start()

        # Wait for started/busy before touching the mic.
        while True:
            try:
                msg = self.inbox.get(timeout=10)
            except queue.Empty:
                return self.fail("daemon did not answer start")
            if msg.get("type") == "started":
                break
            if msg.get("type") == "busy":
                return self.fail(msg.get("message", "daemon busy"))
            if msg.get("type") in ("error", "_disconnected", "_error"):
                return self.fail(msg.get("message", "daemon refused start"))
            # ignore stray partials (none expected this early)

        assert sock is not None
        self.sock = sock

        try:
            mic = spawn_mic(self.sr, int(self.cfg["audio"]["channels"]))
        except RuntimeError as e:
            try:
                proto.send_json(sock, {"type": "cancel",
                                       "request_id": self.rid})
            except OSError:
                pass
            return self.fail(str(e))

        log(f"stream {self.rid}: recording (auto={int(self.auto)})")
        t0 = time.monotonic()
        last_vad_check = 0.0
        end_reason = "release"
        assert mic.stdout is not None

        try:
            while not self.stop_ev.is_set():
                data = mic.stdout.read(self.chunk_bytes)
                if not data:
                    if mic.poll() is not None:
                        end_reason = "mic exited"
                        break
                    continue
                try:
                    proto.send_audio(sock, bytes(data))
                except OSError:
                    end_reason = "socket write failed"
                    break
                self.audio_buf += data
                self.audio_samples += len(data) // 2
                now = time.monotonic()
                if now - t0 >= self.max_hold:
                    end_reason = f"max {self.max_hold:.0f}s reached"
                    break
                if self.auto and now - last_vad_check >= 0.4:
                    last_vad_check = now
                    hit, reason = self._endpoint_reached()
                    if hit:
                        if "no speech" in reason:
                            return self._abort_silence(mic, reason)
                        end_reason = reason
                        break
                # A daemon error mid-stream aborts the recording early.
                # Internal reader statuses (_error/_disconnected) are fatal
                # too: the connection is dead, recording on is pointless.
                # Anything else (e.g. a partial preview) is re-queued, never
                # dropped -- _drain still sees it after STOP.
                try:
                    msg = self.inbox.get_nowait()
                    if msg.get("type") in ("error", "busy", "_error",
                                           "_disconnected"):
                        return self._abort_mic(
                            mic, msg.get("message", "daemon error"))
                    self.inbox.put(msg)
                except queue.Empty:
                    pass
        finally:
            try:
                mic.terminate()
            except OSError:
                pass

        if self.cancel_requested:
            try:
                proto.send_json(sock, {"type": "cancel",
                                       "request_id": self.rid})
            except OSError:
                pass
            self._drain()
            return self.fail("cancelled")
        try:
            proto.send_json(sock, {"type": "stop", "request_id": self.rid})
        except OSError:
            return self.fail("daemon disconnected before stop")
        log(f"stream {self.rid}: stopped ({end_reason}), "
            f"{self.audio_samples / self.sr:.1f}s audio")
        self._drain()
        return self._finish()

    def _abort_mic(self, mic: subprocess.Popen, reason: str) -> int:
        try:
            mic.terminate()
        except OSError:
            pass
        try:
            assert self.sock is not None
            proto.send_json(self.sock, {"type": "cancel",
                                        "request_id": self.rid})
        except OSError:
            pass
        return self.fail(reason)

    def _abort_silence(self, mic: subprocess.Popen, reason: str) -> int:
        """Auto mode heard nothing: don't bother the daemon, empty result."""
        try:
            mic.terminate()
        except OSError:
            pass
        try:
            assert self.sock is not None
            proto.send_json(self.sock, {"type": "cancel",
                                        "request_id": self.rid})
        except OSError:
            pass
        log(f"stream {self.rid}: abort, {reason} (daemon never used)")
        try:
            self._write(self.res_file, "")
        except OSError:
            pass
        return 0

    def _finish(self) -> int:
        msg = self.final or {"type": "_disconnected"}
        mtype = msg.get("type")
        if mtype == "result" and msg.get("request_id") == self.rid:
            text = msg.get("text", "")
            conf = msg.get("confidence")
            low = bool(msg.get("low_confidence", False))
            timings = msg.get("timings", {})
            tag = " LOW-CONF" if low else ""
            log(f"stream {self.rid}: result ({len(text)} chars, "
                f"conf={conf}{tag}) {timings}")
            try:
                self._write(self.res_file, text)
                self._write(self.conf_file,
                            f"{conf} {'low' if low else 'ok'}")
            except OSError as e:
                return self.fail(f"cannot write result ({e})")
            print(text)
            return 0
        if mtype == "cancelled":
            return self.fail("cancelled by daemon")
        return self.fail(msg.get("message", "no result from daemon"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="opencode-voice socket client")
    ap.add_argument("--config", default=None)
    ap.add_argument("--auto", action="store_true",
                    help="VAD endpointing: submit on trailing silence")
    ap.add_argument("--end-silence", type=float, default=1.2)
    ap.add_argument("--no-speech-timeout", type=float, default=12.0)
    ap.add_argument("--max-hold", type=float, default=45.0)
    ap.add_argument("--socket", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    sock_path = args.socket or resolve_socket(cfg)

    st = Streamer(cfg, sock_path, auto=args.auto,
                  end_silence=args.end_silence,
                  no_speech_timeout=args.no_speech_timeout,
                  max_hold=args.max_hold)

    def _sigusr1(signum, frame):
        st.stop_ev.set()  # voice-stop.sh: finalize

    def _cancel(signum, frame):
        st.cancel_requested = True
        st.stop_ev.set()  # SIGTERM/SIGINT: cancel instead of finalize

    signal.signal(signal.SIGUSR1, _sigusr1)
    signal.signal(signal.SIGTERM, _cancel)
    signal.signal(signal.SIGINT, _cancel)
    rc = st.run()
    try:
        if st.sock is not None:
            st.sock.close()
    except OSError:
        pass
    for f in (st.pid_file, st.req_file):
        try:
            os.remove(f)
        except OSError:
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
