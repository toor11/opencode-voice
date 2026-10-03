"""Normalization tests: deterministic, conservative, no model needed."""
import unittest

from config import load as load_config
from normalizer import compile_rules, normalize, rules_from_config


class TestNormalize(unittest.TestCase):
    def setUp(self):
        self.rules = compile_rules([
            ["open code", "OpenCode"],
            ["get hub", "GitHub"],
        ])

    def test_basic_replacement(self):
        text, n = normalize("open code is great", self.rules)
        self.assertEqual(text, "OpenCode is great")
        self.assertEqual(n, 1)

    def test_case_insensitive_match(self):
        text, _ = normalize("Open Code review", self.rules)
        self.assertEqual(text, "OpenCode review")

    def test_word_boundaries(self):
        # Must not fire inside larger words or partial phrases.
        text, n = normalize("reopen code", self.rules)
        self.assertEqual((text, n), ("reopen code", 0))
        text, n = normalize("open codes", self.rules)
        self.assertEqual((text, n), ("open codes", 0))

    def test_ordinary_english_untouched(self):
        for plain in ["open the door", "get hubcaps", "code review",
                      "the open codex", "github actions are great"]:
            text, n = normalize(plain, self.rules)
            self.assertEqual((text, n), (plain, 0), plain)

    def test_multiple_rules(self):
        text, n = normalize("push open code to get hub", self.rules)
        self.assertEqual((text, n), ("push OpenCode to GitHub", 2))

    def test_empty_and_no_rules(self):
        self.assertEqual(normalize("", self.rules), ("", 0))
        self.assertEqual(normalize("open code", []), ("open code", 0))

    def test_deterministic_repeated(self):
        a, na = normalize("open code and get hub", self.rules)
        b, nb = normalize("open code and get hub", self.rules)
        self.assertEqual((a, na), (b, nb))

    def test_dict_form_rules(self):
        rules = compile_rules([{"from": "cue wen", "to": "Qwen"}])
        text, n = normalize("ask cue wen", rules)
        self.assertEqual((text, n), ("ask Qwen", 1))

    def test_flexible_whitespace(self):
        text, n = normalize("open   code", self.rules)
        self.assertEqual((text, n), ("OpenCode", 1))


class TestConfigRules(unittest.TestCase):
    def test_defaults_from_config(self):
        cfg = load_config("/nonexistent-config.toml")
        rules = rules_from_config(cfg)
        self.assertTrue(len(rules) > 5)
        text, n = normalize("commit to get hub with open code", rules)
        self.assertIn("GitHub", text)
        self.assertIn("OpenCode", text)

    def test_disabled_returns_no_rules(self):
        cfg = load_config("/nonexistent-config.toml")
        cfg["normalization"]["enabled"] = False
        self.assertEqual(rules_from_config(cfg), [])

    def test_defaults_are_conservative(self):
        # No single common word may be rewritten by defaults.
        cfg = load_config("/nonexistent-config.toml")
        rules = rules_from_config(cfg)
        for plain in ["rust is fast", "code", "windows update",
                      "python is great", "docker run hello"]:
            text, n = normalize(plain, rules)
            self.assertEqual(n, 0, f"{plain!r} was rewritten to {text!r}")


if __name__ == "__main__":
    unittest.main()
