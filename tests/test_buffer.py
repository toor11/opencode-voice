"""AudioBuffer + ASR scheduling tests: no model, no GPU."""
import threading
import time
import unittest

import numpy as np

from audio_buffer import AudioBuffer


class TestAudioBuffer(unittest.TestCase):
    def test_append_and_full(self):
        buf = AudioBuffer()
        a = np.ones(1600, dtype=np.float32)
        b = np.ones(3200, dtype=np.float32) * 2
        buf.append(a)
        buf.append(b)
        self.assertEqual(len(buf), 4800)
        full = buf.full()
        self.assertEqual(len(full), 4800)
        self.assertTrue((full[:1600] == 1).all())
        self.assertTrue((full[1600:] == 2).all())

    def test_tail_only_copies_what_it_needs(self):
        buf = AudioBuffer()
        for i in range(10):
            buf.append(np.full(1600, i, dtype=np.float32))
        tail = buf.tail(0.2)  # 3200 samples
        self.assertEqual(len(tail), 3200)
        self.assertTrue((tail[:1600] == 8).all())
        self.assertTrue((tail[1600:] == 9).all())

    def test_tail_more_than_buffered(self):
        buf = AudioBuffer()
        buf.append(np.ones(100, dtype=np.float32))
        self.assertEqual(len(buf.tail(10.0)), 100)

    def test_empty(self):
        buf = AudioBuffer()
        self.assertEqual(len(buf), 0)
        self.assertEqual(len(buf.full()), 0)
        self.assertEqual(len(buf.tail(1.0)), 0)
        self.assertEqual(buf.seconds, 0.0)

    def test_bounded_memory(self):
        buf = AudioBuffer(max_seconds=1.0)
        chunk = np.ones(16000, dtype=np.float32)
        for _ in range(5):
            buf.append(chunk)
        self.assertLessEqual(len(buf), 16000)
        self.assertLessEqual(buf.seconds, 1.0)

    def test_thread_safe_appends(self):
        buf = AudioBuffer()
        chunk = np.ones(100, dtype=np.float32)

        def add():
            for _ in range(100):
                buf.append(chunk)

        threads = [threading.Thread(target=add) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(buf), 40000)
        self.assertEqual(len(buf.full()), 40000)


if __name__ == "__main__":
    unittest.main()
