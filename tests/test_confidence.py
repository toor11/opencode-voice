"""Confidence scoring tests: pure math, no model, no GPU."""
import math
import unittest

from asr import aggregate_confidence, segment_confidence, segment_info


class Seg:
    def __init__(self, text="hi", avg_logprob=-0.2, no_speech_prob=0.05,
                 start=0.0, end=1.0):
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.start = start
        self.end = end


class TestSegmentConfidence(unittest.TestCase):
    def test_good_logprob_high_confidence(self):
        self.assertAlmostEqual(segment_confidence(Seg(avg_logprob=-0.1)),
                               math.exp(-0.1), places=3)

    def test_bad_logprob_low_confidence(self):
        self.assertLess(segment_confidence(Seg(avg_logprob=-1.5)), 0.25)

    def test_perfect_is_one(self):
        self.assertEqual(segment_confidence(Seg(avg_logprob=0.0)), 1.0)

    def test_missing_data_is_none_not_zero(self):
        class Bare:
            pass
        self.assertIsNone(segment_confidence(Bare()))

    def test_nan_is_none(self):
        self.assertIsNone(segment_confidence(Seg(avg_logprob=float("nan"))))

    def test_none_is_none(self):
        self.assertIsNone(segment_confidence(Seg(avg_logprob=None)))


class TestAggregate(unittest.TestCase):
    def test_duration_weighted(self):
        # 1s at ~1.0 and 3s at ~0.5 -> closer to 0.5 than to 1.0
        segs = [Seg(avg_logprob=0.0, start=0, end=1),
                Seg(avg_logprob=math.log(0.5), start=1, end=4)]
        conf = aggregate_confidence(segs)
        self.assertAlmostEqual(conf, (1.0 + 3 * 0.5) / 4, places=3)

    def test_empty_is_none(self):
        self.assertIsNone(aggregate_confidence([]))

    def test_all_missing_is_none(self):
        class Bare:
            start = 0.0
            end = 1.0
        self.assertIsNone(aggregate_confidence([Bare()]))

    def test_skips_bad_segments_keeps_good(self):
        class Bare:
            start = 0.0
            end = 1.0
        conf = aggregate_confidence([Bare(), Seg(avg_logprob=0.0)])
        self.assertEqual(conf, 1.0)

    def test_segment_info_shape(self):
        info = segment_info(Seg(text="open", avg_logprob=-0.2,
                                no_speech_prob=0.1, start=0.5, end=1.5))
        self.assertEqual(info["text"], "open")
        self.assertEqual(info["start"], 0.5)
        self.assertIn("confidence", info)
        self.assertIn("no_speech_prob", info)


if __name__ == "__main__":
    unittest.main()
