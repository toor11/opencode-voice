"""Protocol framing tests: no model, no socket server, no GPU needed."""
import socket
import struct
import threading
import unittest

import proto
from proto import (
    FRAME_AUDIO,
    FRAME_JSON,
    ConnClosed,
    ProtocolError,
    encode_frame,
    loads,
    new_request_id,
    read_frame,
    read_json,
    s16_bytes_to_float32,
    send_audio,
    send_json,
)


def socketpair():
    a, b = socket.socketpair()
    return a, b


class TestFrames(unittest.TestCase):
    def test_json_roundtrip(self):
        a, b = socketpair()
        try:
            send_json(a, {"type": "start", "request_id": "r1"})
            kind, payload = read_frame(b)
            self.assertEqual(kind, FRAME_JSON)
            self.assertEqual(loads(payload),
                             {"type": "start", "request_id": "r1"})
        finally:
            a.close()
            b.close()

    def test_audio_roundtrip(self):
        a, b = socketpair()
        try:
            send_audio(a, b"\x01\x02" * 800)
            kind, payload = read_frame(b)
            self.assertEqual(kind, FRAME_AUDIO)
            self.assertEqual(payload, b"\x01\x02" * 800)
        finally:
            a.close()
            b.close()

    def test_partial_reads_and_coalescing(self):
        """1-byte writes must reassemble; back-to-back frames must split."""
        a, b = socketpair()
        try:
            blob = (encode_frame(FRAME_JSON, b'{"type":"stop"}')
                    + encode_frame(FRAME_AUDIO, b"\x00" * 100))
            for i in range(0, len(blob), 1):
                a.sendall(blob[i:i + 1])
            k1, p1 = read_frame(b)
            k2, p2 = read_frame(b)
            self.assertEqual((k1, p1), (FRAME_JSON, b'{"type":"stop"}'))
            self.assertEqual((k2, p2), (FRAME_AUDIO, b"\x00" * 100))
        finally:
            a.close()
            b.close()

    def test_unknown_kind_rejected(self):
        a, b = socketpair()
        try:
            a.sendall(struct.pack("!BI", 0x7F, 3) + b"abc")
            with self.assertRaises(ProtocolError):
                read_frame(b)
        finally:
            a.close()
            b.close()

    def test_huge_length_rejected_without_reading(self):
        a, b = socketpair()
        try:
            a.sendall(struct.pack("!BI", FRAME_JSON, 1 << 30))
            with self.assertRaises(ProtocolError):
                read_frame(b)
        finally:
            a.close()
            b.close()

    def test_bad_json_rejected(self):
        a, b = socketpair()
        try:
            a.sendall(encode_frame(FRAME_JSON, b"not json{"))
            with self.assertRaises(ProtocolError):
                read_json(b)
        finally:
            a.close()
            b.close()

    def test_json_without_type_rejected(self):
        a, b = socketpair()
        try:
            send_json(a, {"hello": 1})
            with self.assertRaises(ProtocolError):
                read_json(b)
        finally:
            a.close()
            b.close()

    def test_audio_frame_where_json_expected(self):
        a, b = socketpair()
        try:
            send_audio(a, b"\x00\x00")
            with self.assertRaises(ProtocolError):
                read_json(b)
        finally:
            a.close()
            b.close()

    def test_closed_connection_raises(self):
        a, b = socketpair()
        a.close()
        with self.assertRaises(ConnClosed):
            read_frame(b)
        b.close()

    def test_half_frame_then_close(self):
        a, b = socketpair()
        try:
            a.sendall(struct.pack("!BI", FRAME_JSON, 100) + b"short")
            a.close()
            with self.assertRaises(ConnClosed):
                read_frame(b)
        finally:
            b.close()

    def test_send_after_close_raises_oserror(self):
        a, b = socketpair()
        a.close()
        b.close()
        with self.assertRaises(OSError):
            send_json(a, {"type": "stop"})


class TestAudioConvert(unittest.TestCase):
    def test_s16_roundtrip_scale(self):
        import numpy as np

        raw = (np.array([0, 32767, -32768, 16384], dtype=np.int16)).tobytes()
        out = s16_bytes_to_float32(raw)
        self.assertAlmostEqual(out[0], 0.0)
        self.assertAlmostEqual(out[1], 32767 / 32768.0)
        self.assertAlmostEqual(out[2], -1.0)
        self.assertAlmostEqual(out[3], 0.5)

    def test_empty(self):
        self.assertEqual(len(s16_bytes_to_float32(b"")), 0)

    def test_odd_length_tolerated(self):
        out = s16_bytes_to_float32(b"\x00\x00\xff")
        self.assertEqual(len(out), 1)


class TestRequestIds(unittest.TestCase):
    def test_unique_and_shaped(self):
        ids = {new_request_id() for _ in range(200)}
        self.assertEqual(len(ids), 200)
        for rid in ids:
            date, clock, rand = rid.split("-")
            self.assertEqual(len(date), 8)
            self.assertEqual(len(clock), 6)
            self.assertEqual(len(rand), 8)

    def test_concurrent_generation_unique(self):
        out, lock = [], threading.Lock()

        def gen():
            local = [new_request_id() for _ in range(50)]
            with lock:
                out.extend(local)

        threads = [threading.Thread(target=gen) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(set(out)), 400)


if __name__ == "__main__":
    unittest.main()
