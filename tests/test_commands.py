"""Tests for voice-cmd.sh: safe command registry + deterministic parser.

Runs the real script with OC_VOICE_DRYRUN=1 (parser only, never launches)
plus non-dryrun injection cases (must exit 1 having executed nothing).
Hermetic w.r.t. installed apps: PATH is overridden with stub dirs so
INSTALLED yes/no answers are deterministic on any machine.
"""
import os
import stat
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CMD = os.path.join(REPO, "voice-cmd.sh")


def make_stubs(names):
    """Hermetic bin dir: stub apps + links to the coreutils the script needs
    (tr/sed/date), so PATH can exclude /usr/bin and exe lookups stay
    deterministic on any machine."""
    d = tempfile.mkdtemp(prefix="oc-voice-stubs-")
    for tool in ("tr", "sed", "date"):
        src = "/usr/bin/" + tool
        if os.path.exists(src):
            os.symlink(src, os.path.join(d, tool))
    for n in names:
        p = os.path.join(d, n)
        with open(p, "w") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return d


STUBS_FULL = make_stubs(["firedragon", "brave", "chromium", "code",
                          "kitty", "hyprctl", "notify-send"])
STUBS_EMPTY = make_stubs([])


def run(text, dryrun=True, path=None):
    env = dict(os.environ)
    env["OC_VOICE_DRYRUN"] = "1" if dryrun else "0"
    env["PATH"] = path if path is not None else STUBS_FULL
    p = subprocess.run([CMD, text], capture_output=True, text=True,
                       timeout=30, env=env)
    fields = {}
    for line in p.stdout.splitlines():
        if ": " in line:
            k, v = line.split(": ", 1)
            fields[k.strip()] = v.strip()
    return p.returncode, fields


class TestValidCommands(unittest.TestCase):
    def test_open_firefox(self):
        rc, f = run("open firefox")
        self.assertEqual(rc, 0)
        self.assertEqual(f.get("COMMAND"), "launch_app")
        self.assertEqual(f.get("TARGET"), "firedragon")
        self.assertEqual(f.get("EXECUTABLE"), "firedragon")
        self.assertEqual(f.get("INSTALLED"), "yes")

    def test_asr_variants(self):
        for text in ("OpenFire Fox.", "OpenFire folks.", "open fire fox",
                     "start firefox browser", "launch firedragon",
                     "open fire dragon"):
            rc, f = run(text)
            self.assertEqual(rc, 0, text)
            self.assertEqual(f.get("TARGET"), "firedragon", text)

    def test_brave_chromium(self):
        for text, target in (("open brave", "brave"),
                             ("launch brave", "brave"),
                             ("open brave browser", "brave"),
                             ("open chromium", "chromium"),
                             ("run chromium", "chromium"),
                             ("please open brave", "brave"),
                             ("open the brave", "brave")):
            rc, f = run(text)
            self.assertEqual(rc, 0, text)
            self.assertEqual(f.get("TARGET"), target, text)

    def test_opencode(self):
        for text in ("launch opencode",
                     "open opencode and start working"):
            rc, f = run(text)
            self.assertEqual(rc, 0, text)
            self.assertEqual(f.get("COMMAND"), "opencode.work", text)
            self.assertIn("opencode", f.get("EXECUTABLE", ""), text)

    def test_watch_session(self):
        for text in ("show session",
                     "show opencode session",
                     "open session"):
            rc, f = run(text)
            self.assertEqual(rc, 0, text)
            self.assertEqual(f.get("COMMAND"), "watch_session", text)
            self.assertTrue(f.get("SCRIPT", "").endswith("voice-watch.sh"),
                            text)

    def test_missing_app_reports_not_installed(self):
        rc, f = run("open brave", path=STUBS_EMPTY)
        self.assertEqual(rc, 0)
        self.assertEqual(f.get("TARGET"), "brave")
        self.assertEqual(f.get("INSTALLED"), "no")


class TestNonCommands(unittest.TestCase):
    def test_ordinary_speech_falls_through(self):
        for text in ("tell me about firefox",
                     "fix firefox",
                     "search for firefox",
                     "why is firefox slow",
                     "open source software",
                     "I am working on OpenCode",
                     "what should I do with chromium",
                     "create a Firefox extension",
                     "open the Firefox configuration file",
                     "open final file",
                     "opening brave",
                     "opencode is great"):
            rc, f = run(text)
            self.assertEqual(rc, 1, text)
            self.assertNotIn("COMMAND", f, text)


class TestSecurity(unittest.TestCase):
    PWNED = "/tmp/pwned-voice-cmd-test"

    def setUp(self):
        try:
            os.remove(self.PWNED)
        except OSError:
            pass

    def tearDown(self):
        try:
            os.remove(self.PWNED)
        except OSError:
            pass

    def test_injection_strings_never_execute(self):
        for text in ("open firefox; touch " + self.PWNED,
                     "open firefox && touch " + self.PWNED,
                     "open firefox | touch " + self.PWNED,
                     "open $(touch " + self.PWNED + ")",
                     "open `touch " + self.PWNED + "`",
                     "$(touch " + self.PWNED + ")",
                     "; rm -rf /tmp/oc-voice-test-dir"):
            rc, f = run(text, dryrun=False)  # real mode, must still do nothing
            self.assertEqual(rc, 1, text)
            self.assertNotIn("COMMAND", f, text)
        self.assertFalse(os.path.exists(self.PWNED),
                         "injection executed a shell command")

    def test_executable_never_comes_from_transcript(self):
        rc, f = run("open firefox", dryrun=True)
        self.assertEqual(f.get("EXECUTABLE"), "firedragon")
        # The transcript word "firefox" must not appear as the executable.
        self.assertNotEqual(f.get("EXECUTABLE"), "firefox")


if __name__ == "__main__":
    unittest.main()
