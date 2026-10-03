"""Server tests with a mocked Whisper model: run without CUDA/GPU.

Covers: START/AUDIO/STOP, CANCEL, partials, busy, silence (no ASR call),
empty recording, malformed messages, disconnects, request_id isolation and
reconnect. VAD itself runs for real (onnxruntime CPU).
"""
import os
import shutil
import socket
import tempfile
import threading
import time
import unittest

import numpy as np

import proto
from config import DEFAULTS, load as load_config

import importlib.util
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "voice_daemon", os.path.join(_HERE, "..", "voice-daemon.py"))
assert _SPEC and _SPEC.loader
voice_daemon = importlib.util.module_from_spec(_SPEC)
sys.modules["voice_daemon"] = voice_daemon
_SPEC.loader.exec_module(voice_daemon)
from voice_daemon import VoiceServer  # noqa: E402


class FakeSegment:
    def __init__(self, text, avg_logprob=-0.2, no_speech_prob=0.05,
                 start=0.0, end=1.0):
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.start = start
        self.end = end


class FakeModel:
    """Stands in for WhisperModel. Records every transcribe() call."""

    def __init__(self, text="hello world", delay=0.0, avg_logprob=-0.2):
        self.text = text
        self.delay = delay
        self.avg_logprob = avg_logprob
        self.calls = []
        self.lock = threading.Lock()

    def transcribe(self, audio, **kwargs):
        import numpy as _np

        if isinstance(audio, _np.ndarray):
            dur = len(audio) / 16000
        else:
            dur = -1.0
        with self.lock:
            self.calls.append({"duration_s": dur, "kwargs": kwargs})
        if self.delay:
            time.sleep(self.delay)
        seg = FakeSegment(self.text, avg_logprob=self.avg_logprob,
                          start=0.0, end=max(dur, 0.1))
        return [seg], object()


def make_sine(seconds=1.0, freq=440.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    wave = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    return wave.tobytes()


def make_silence(seconds=1.0, sr=16000):
    return (np.zeros(int(seconds * sr), dtype=np.int16)).tobytes()


def test_config(**over):
    cfg = load_config("/nonexistent-config.toml")
    cfg["streaming"]["enabled"] = True
    cfg["streaming"]["interval_ms"] = 200
    cfg["streaming"]["min_new_speech_s"] = 0.3
    cfg["streaming"]["window_s"] = 4.0
    for section, values in over.items():
        cfg[section].update(values)
    return cfg


class ServerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="oc-voice-test-")
        self.sock = os.path.join(self.tmp, "voice.sock")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def start_server(self, model, **over):
        cfg = test_config(**over)
        server = VoiceServer(cfg, model, socket_path=self.sock)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        for _ in range(100):
            if os.path.exists(self.sock):
                break
            time.sleep(0.05)
        self.server = server
        self.thread = thread
        return server

    def stop_server(self):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.assertFalse(os.path.exists(self.sock),
                         "socket file must be cleaned up")

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.sock)
        s.settimeout(10)
        return s


class TestBasicFlow(ServerCase):
    def test_stop_returns_result_with_timings(self):
        model = FakeModel("open the authentication file")
        self.start_server(model, vad={"enabled": False})
        s = self.connect()
        try:
            rid = "20261003-170000-aaa111"
            proto.send_json(s, {"type": "start", "request_id": rid,
                                "sample_rate": 16000})
            msg = proto.read_json(s)
            self.assertEqual(msg, {"type": "started", "request_id": rid})
            proto.send_audio(s, make_sine(0.5))
            proto.send_audio(s, make_sine(0.5))
            proto.send_json(s, {"type": "stop", "request_id": rid})
            final = None
            while True:
                m = proto.read_json(s)
                if m["type"] == "partial":
                    self.assertEqual(m["request_id"], rid)  # replaces, not appends
                elif m["type"] == "result":
                    final = m
                    break
                else:
                    self.fail(f"unexpected {m}")
            self.assertEqual(final["request_id"], rid)
            self.assertEqual(final["text"], "open the authentication file")
            for key in ("duration_s", "vad_s", "asr_s", "finalization_s",
                        "total_s", "partials"):
                self.assertIn(key, final["timings"])
            self.assertGreaterEqual(len(model.calls), 1)
        finally:
            s.close()
            self.stop_server()

    def test_empty_recording_returns_empty_without_asr(self):
        model = FakeModel()
        self.start_server(model)
        s = self.connect()
        try:
            proto.send_json(s, {"type": "start", "request_id": "e1"})
            proto.read_json(s)
            proto.send_json(s, {"type": "stop", "request_id": "e1"})
            m = proto.read_json(s)
            self.assertEqual(m["type"], "result")
            self.assertEqual(m["text"], "")
            self.assertEqual(len(model.calls), 0)
        finally:
            s.close()
            self.stop_server()

    def test_silence_skips_asr(self):
        model = FakeModel()
        self.start_server(model)  # VAD enabled by default
        s = self.connect()
        try:
            proto.send_json(s, {"type": "start", "request_id": "s1"})
            proto.read_json(s)
            for _ in range(5):
                proto.send_audio(s, make_silence(0.4))
            proto.send_json(s, {"type": "stop", "request_id": "s1"})
            seen_partial = False
            while True:
                m = proto.read_json(s)
                if m["type"] == "partial":
                    seen_partial = True
                elif m["type"] == "result":
                    break
            self.assertFalse(seen_partial, "silence must not emit partials")
            self.assertEqual(m["text"], "")
            asr_calls = [c for c in model.calls]
            self.assertEqual(len(asr_calls), 0,
                             "silence must never reach the model")
        finally:
            s.close()
            self.stop_server()

    def test_cancel_stops_processing(self):
        model = FakeModel(delay=0.5)
        self.start_server(model, vad={"enabled": False})
        s = self.connect()
        try:
            proto.send_json(s, {"type": "start", "request_id": "c1"})
            proto.read_json(s)
            proto.send_audio(s, make_sine(0.5))
            proto.send_json(s, {"type": "cancel", "request_id": "c1"})
            m = proto.read_json(s)
            self.assertEqual(m, {"type": "cancelled", "request_id": "c1"})
        finally:
            s.close()
            self.stop_server()

    def test_partials_arrive_before_stop(self):
        model = FakeModel("open the auth")
        self.start_server(model, vad={"enabled": False})
        s = self.connect()
        try:
            proto.send_json(s, {"type": "start", "request_id": "p1"})
            proto.read_json(s)
            # Pace chunks slower than the 200ms interval so partials trigger.
            for _ in range(4):
                proto.send_audio(s, make_sine(0.2))
                time.sleep(0.35)
            got_partial = []

            def collector():
                try:
                    while True:
                        m = proto.read_json(s)
                        if m["type"] == "partial":
                            got_partial.append(m["text"])
                        elif m["type"] == "result":
                            got_partial.append("RESULT:" + m["text"])
                            return
                except Exception:
                    return

            t = threading.Thread(target=collector, daemon=True)
            t.start()
            for _ in range(4):
                proto.send_audio(s, make_sine(0.2))
                time.sleep(0.35)
            proto.send_json(s, {"type": "stop", "request_id": "p1"})
            t.join(timeout=15)
            self.assertTrue(any(x == "open the auth" for x in got_partial),
                            f"expected a partial, got {got_partial}")
            self.assertIn("RESULT:open the auth", got_partial)
        finally:
            s.close()
            self.stop_server()


class TestRobustness(ServerCase):
    def test_second_session_gets_busy(self):
        model = FakeModel(delay=0.3)
        self.start_server(model, vad={"enabled": False})
        a = self.connect()
        try:
            proto.send_json(a, {"type": "start", "request_id": "busy-a"})
            self.assertEqual(proto.read_json(a)["type"], "started")
            b = self.connect()
            try:
                proto.send_json(b, {"type": "start", "request_id": "busy-b"})
                m = proto.read_json(b)
                self.assertEqual(m["type"], "busy")
            finally:
                b.close()
            proto.send_json(a, {"type": "cancel", "request_id": "busy-a"})
            self.assertEqual(proto.read_json(a)["type"], "cancelled")
        finally:
            a.close()
        # Server still healthy afterwards.
        c = self.connect()
        try:
            proto.send_json(c, {"type": "start", "request_id": "after"})
            self.assertEqual(proto.read_json(c)["type"], "started")
            proto.send_json(c, {"type": "stop", "request_id": "after"})
            m = proto.read_json(c)
            self.assertEqual(m["type"], "result")
        finally:
            c.close()
            self.stop_server()

    def test_malformed_first_message(self):
        self.start_server(FakeModel())
        s = self.connect()
        try:
            proto.send_json(s, {"type": "nonsense"})
            m = proto.read_json(s)
            self.assertEqual(m["type"], "error")
        finally:
            s.close()
            self.stop_server()

    def test_garbage_bytes_close_cleanly_then_reconnect(self):
        self.start_server(FakeModel())
        s = self.connect()
        try:
            s.sendall(b"\xff\xff\xff\xffgarbage!!!")
            try:
                proto.read_json(s)
            except (proto.ProtocolError, proto.ConnClosed, OSError):
                pass
        finally:
            s.close()
        time.sleep(0.3)
        c = self.connect()
        try:
            proto.send_json(c, {"type": "start", "request_id": "re1"})
            self.assertEqual(proto.read_json(c)["type"], "started")
            proto.send_json(c, {"type": "stop", "request_id": "re1"})
            self.assertEqual(proto.read_json(c)["type"], "result")
        finally:
            c.close()
            self.stop_server()

    def test_request_id_mismatch_rejected(self):
        self.start_server(FakeModel())
        s = self.connect()
        try:
            proto.send_json(s, {"type": "start", "request_id": "good"})
            proto.read_json(s)
            proto.send_json(s, {"type": "stop", "request_id": "evil"})
            m = proto.read_json(s)
            self.assertEqual(m["type"], "error")
        finally:
            s.close()
            self.stop_server()

    def test_client_crash_mid_stream_frees_server(self):
        model = FakeModel()
        self.start_server(model, vad={"enabled": False})
        s = self.connect()
        proto.send_json(s, {"type": "start", "request_id": "crash"})
        proto.read_json(s)
        proto.send_audio(s, make_sine(0.3))
        s.close()  # crash: no stop, no cancel
        time.sleep(0.5)
        c = self.connect()
        try:
            proto.send_json(c, {"type": "start", "request_id": "next"})
            self.assertEqual(proto.read_json(c)["type"], "started")
            proto.send_json(c, {"type": "stop", "request_id": "next"})
            self.assertEqual(proto.read_json(c)["type"], "result")
        finally:
            c.close()
            self.stop_server()

    def test_wrong_sample_rate_rejected(self):
        self.start_server(FakeModel())
        s = self.connect()
        try:
            proto.send_json(s, {"type": "start", "request_id": "sr",
                                "sample_rate": 44100})
            m = proto.read_json(s)
            self.assertEqual(m["type"], "error")
        finally:
            s.close()
            self.stop_server()

    def test_asr_failure_reports_error_not_hang(self):
        class BoomModel(FakeModel):
            def transcribe(self, audio, **kwargs):
                raise RuntimeError("simulated CUDA failure")

        self.start_server(BoomModel(), vad={"enabled": False},
                          streaming={"enabled": False})
        s = self.connect()
        try:
            proto.send_json(s, {"type": "start", "request_id": "boom"})
            proto.read_json(s)
            proto.send_audio(s, make_sine(0.4))
            proto.send_json(s, {"type": "stop", "request_id": "boom"})
            s.settimeout(15)
            m = proto.read_json(s)
            self.assertEqual(m["type"], "error")
        finally:
            s.close()
            self.stop_server()

    def test_stale_socket_file_reused(self):
        # Simulate a crashed daemon: dead file at the socket path.
        with open(self.sock, "w") as f:
            f.write("stale")
        self.start_server(FakeModel())
        self.stop_server()


class TestConfidenceAndNormalization(ServerCase):
    def test_result_carries_confidence(self):
        import math

        model = FakeModel("hello world", avg_logprob=-0.2)
        self.start_server(model, vad={"enabled": False})
        s = self.connect()
        try:
            proto.send_json(s, {"type": "start", "request_id": "cf1"})
            proto.read_json(s)
            proto.send_audio(s, make_sine(0.5))
            proto.send_json(s, {"type": "stop", "request_id": "cf1"})
            while True:
                m = proto.read_json(s)
                if m["type"] == "result":
                    break
            self.assertAlmostEqual(m["confidence"], math.exp(-0.2), places=3)
            self.assertFalse(m["low_confidence"])
            self.assertIn("confidence", m["timings"])
        finally:
            s.close()
            self.stop_server()

    def test_low_confidence_flagged_not_dropped(self):
        model = FakeModel("mumble", avg_logprob=-1.5)
        self.start_server(model, vad={"enabled": False})
        s = self.connect()
        try:
            proto.send_json(s, {"type": "start", "request_id": "cf2"})
            proto.read_json(s)
            proto.send_audio(s, make_sine(0.5))
            proto.send_json(s, {"type": "stop", "request_id": "cf2"})
            while True:
                m = proto.read_json(s)
                if m["type"] == "result":
                    break
            self.assertTrue(m["low_confidence"])
            self.assertEqual(m["text"], "mumble")  # still returned
        finally:
            s.close()
            self.stop_server()

    def test_normalization_applied_to_result(self):
        model = FakeModel("open code on get hub")
        self.start_server(model, vad={"enabled": False})
        s = self.connect()
        try:
            proto.send_json(s, {"type": "start", "request_id": "nm1"})
            proto.read_json(s)
            proto.send_audio(s, make_sine(0.5))
            proto.send_json(s, {"type": "stop", "request_id": "nm1"})
            while True:
                m = proto.read_json(s)
                if m["type"] == "result":
                    break
            self.assertEqual(m["text"], "OpenCode on GitHub")
        finally:
            s.close()
            self.stop_server()

    def test_normalization_disabled(self):
        model = FakeModel("open code on get hub")
        self.start_server(model, vad={"enabled": False},
                          normalization={"enabled": False})
        s = self.connect()
        try:
            proto.send_json(s, {"type": "start", "request_id": "nm2"})
            proto.read_json(s)
            proto.send_audio(s, make_sine(0.5))
            proto.send_json(s, {"type": "stop", "request_id": "nm2"})
            while True:
                m = proto.read_json(s)
                if m["type"] == "result":
                    break
            self.assertEqual(m["text"], "open code on get hub")
        finally:
            s.close()
            self.stop_server()


class TestScheduling(ServerCase):
    def test_final_only_default_single_inference(self):
        # New default: streaming disabled -> no partials, exactly ONE
        # model call per utterance (plus CPU VAD, which is not inference).
        model = FakeModel("one shot")
        self.start_server(model, vad={"enabled": False},
                          streaming={"enabled": False})
        s = self.connect()
        try:
            proto.send_json(s, {"type": "start", "request_id": "fo1"})
            proto.read_json(s)
            for _ in range(6):
                proto.send_audio(s, make_sine(0.2))
                time.sleep(0.15)
            proto.send_json(s, {"type": "stop", "request_id": "fo1"})
            partials = 0
            while True:
                m = proto.read_json(s)
                if m["type"] == "partial":
                    partials += 1
                elif m["type"] == "result":
                    break
            self.assertEqual(partials, 0)
            self.assertEqual(m["text"], "one shot")
            self.assertEqual(len(model.calls), 1)
            self.assertFalse(m["timings"]["partials_enabled"])
        finally:
            s.close()
            self.stop_server()

    def test_no_overlapping_asr(self):
        # Slow model + fast cadence: partials must skip, never overlap.
        import threading as _t

        class CountingModel(FakeModel):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.live = 0
                self.peak = 0
                self.plock = _t.Lock()

            def transcribe(self, audio, **kwargs):
                with self.plock:
                    self.live += 1
                    self.peak = max(self.peak, self.live)
                try:
                    return super().transcribe(audio, **kwargs)
                finally:
                    with self.plock:
                        self.live -= 1

        model = CountingModel("slow", delay=0.6)
        self.start_server(model, vad={"enabled": False})
        s = self.connect()
        try:
            proto.send_json(s, {"type": "start", "request_id": "ov1"})
            proto.read_json(s)
            for _ in range(8):
                proto.send_audio(s, make_sine(0.2))
                time.sleep(0.15)
            proto.send_json(s, {"type": "stop", "request_id": "ov1"})
            while True:
                m = proto.read_json(s)
                if m["type"] == "result":
                    break
            self.assertEqual(m["text"], "slow")
            self.assertEqual(model.peak, 1,
                             "ASR calls overlapped on one model")
        finally:
            s.close()
            self.stop_server()

    def test_stale_partial_never_leaks_into_next_session(self):
        # Cancel while a partial is in-flight, then verify the next
        # session only ever sees its own text.
        model = FakeModel("stale-words", delay=1.0)
        self.start_server(model, vad={"enabled": False})
        a = self.connect()
        try:
            proto.send_json(a, {"type": "start", "request_id": "STALE-A"})
            proto.read_json(a)
            for _ in range(4):
                proto.send_audio(a, make_sine(0.2))
                time.sleep(0.3)
            # A partial ASR job is now likely in-flight; cancel it.
            proto.send_json(a, {"type": "cancel", "request_id": "STALE-A"})
            got = proto.read_json(a)
            self.assertEqual(got["type"], "cancelled")
            a.settimeout(2.5)
            try:
                late = proto.read_json(a)
                self.assertNotIn("stale-words", str(late),
                                 f"stale output leaked: {late}")
            except (proto.ConnClosed, socket.timeout, OSError):
                pass
        finally:
            a.close()
        model.text = "fresh-words"
        b = self.connect()
        try:
            proto.send_json(b, {"type": "start", "request_id": "FRESH-B"})
            proto.read_json(b)
            proto.send_audio(b, make_sine(0.4))
            proto.send_json(b, {"type": "stop", "request_id": "FRESH-B"})
            texts = []
            while True:
                m = proto.read_json(b)
                if m["type"] in ("partial", "result"):
                    texts.append(m["text"])
                if m["type"] == "result":
                    break
            self.assertTrue(all("stale-words" not in t for t in texts))
            self.assertEqual(m["text"], "fresh-words")
        finally:
            b.close()
            self.stop_server()

    def test_rapid_stop_start(self):
        model = FakeModel("quick")
        self.start_server(model, vad={"enabled": False},
                          streaming={"enabled": False})
        try:
            for i in range(3):
                s = self.connect()
                rid = f"rapid-{i}"
                proto.send_json(s, {"type": "start", "request_id": rid})
                proto.read_json(s)
                proto.send_audio(s, make_sine(0.3))
                proto.send_json(s, {"type": "stop", "request_id": rid})
                while True:
                    m = proto.read_json(s)
                    if m["type"] == "result":
                        break
                s.close()
                self.assertEqual(m["request_id"], rid)
                self.assertEqual(m["text"], "quick")
        finally:
            self.stop_server()


class TestCancellation(ServerCase):
    def test_cancel_during_blocking_asr_reacts_fast(self):
        # Model blocks 5s per call: CANCELLED must still arrive promptly,
        # and no RESULT may follow on that connection.
        model = FakeModel("too late", delay=5.0)
        self.start_server(model, vad={"enabled": False})
        s = self.connect()
        try:
            proto.send_json(s, {"type": "start", "request_id": "cx1"})
            proto.read_json(s)
            proto.send_audio(s, make_sine(0.5))
            proto.send_json(s, {"type": "stop", "request_id": "cx1"})
            time.sleep(0.5)  # let the worker enter blocking transcribe
            t0 = time.monotonic()
            proto.send_json(s, {"type": "cancel", "request_id": "cx1"})
            m = proto.read_json(s)
            dt = time.monotonic() - t0
            self.assertEqual(m["type"], "cancelled")
            self.assertLess(dt, 2.0,
                            f"cancel took {dt:.2f}s despite blocking ASR")
            # The abandoned worker must never deliver a late result here.
            s.settimeout(6.0)
            try:
                late = proto.read_json(s)
                self.fail(f"late message after cancel: {late}")
            except (proto.ConnClosed, socket.timeout, OSError):
                pass
        finally:
            s.close()
            self.stop_server()

    def test_cancelled_request_never_reaches_next_request(self):
        model = FakeModel("from-a", delay=1.0)
        self.start_server(model, vad={"enabled": False})
        a = self.connect()
        try:
            proto.send_json(a, {"type": "start", "request_id": "A"})
            proto.read_json(a)
            proto.send_audio(a, make_sine(0.4))
            proto.send_json(a, {"type": "stop", "request_id": "A"})
            time.sleep(0.3)
            proto.send_json(a, {"type": "cancel", "request_id": "A"})
            self.assertEqual(proto.read_json(a)["type"], "cancelled")
        finally:
            a.close()
        time.sleep(1.5)  # let A's abandoned ASR finish in the background
        model.text = "from-b"
        b = self.connect()
        try:
            proto.send_json(b, {"type": "start", "request_id": "B"})
            proto.read_json(b)
            proto.send_audio(b, make_sine(0.4))
            proto.send_json(b, {"type": "stop", "request_id": "B"})
            while True:
                m = proto.read_json(b)
                if m["type"] == "result":
                    break
            self.assertEqual(m["request_id"], "B")
            self.assertEqual(m["text"], "from-b")
        finally:
            b.close()
            self.stop_server()


class TestVadGate(unittest.TestCase):
    def test_silence_array_has_no_speech(self):
        from vad_gate import analyze_array

        audio = np.zeros(16000, dtype=np.float32)
        res = analyze_array(audio)
        self.assertFalse(res.has_speech)
        self.assertEqual(res.segments, [])

    def test_options_from_config(self):
        from vad_gate import options_from_config

        cfg = load_config("/nonexistent-config.toml")
        cfg["vad"]["threshold"] = 0.7
        opts = options_from_config(cfg)
        self.assertEqual(opts.threshold, 0.7)
        self.assertEqual(opts.min_silence_duration_ms,
                         DEFAULTS["vad"]["min_silence_duration_ms"])


class TestConfig(unittest.TestCase):
    def test_resolve_socket_xdg(self):
        import config as config_mod

        tmp = tempfile.mkdtemp()
        try:
            os.environ["XDG_RUNTIME_DIR"] = tmp
            cfg = load_config("/nonexistent-config.toml")
            self.assertEqual(config_mod.resolve_socket(cfg),
                             os.path.join(tmp, "opencode-voice.sock"))
        finally:
            del os.environ["XDG_RUNTIME_DIR"]
            shutil.rmtree(tmp, ignore_errors=True)

    def test_resolve_socket_fallback(self):
        import config as config_mod

        os.environ.pop("XDG_RUNTIME_DIR", None)
        cfg = load_config("/nonexistent-config.toml")
        self.assertTrue(config_mod.resolve_socket(cfg).endswith("voice.sock"))

    def test_env_override(self):
        os.environ["OC_VOICE_MODEL"] = "tiny.en"
        try:
            cfg = load_config("/nonexistent-config.toml")
            self.assertEqual(cfg["transcription"]["model"], "tiny.en")
        finally:
            del os.environ["OC_VOICE_MODEL"]


if __name__ == "__main__":
    unittest.main()
