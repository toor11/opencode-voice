# Contributing to opencode-voice

First, thank you for your interest in contributing! This guide will help you get the project set up and ready to contribute.

## Table of Contents
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Requests](#pull-requests)
- [Code Style](#code-style)

## Quick Start

### Prerequisites
- Python 3.12+
- Hyprland/Wayland
- Git

### Clone and Setup
```bash
# Clone the repository
git clone https://github.com/toor11/opencode-voice.git
cd opencode-voice

# Create a Python virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install --python .venv/bin/python faster-whisper piper-tts soundfile onnxruntime "av==12.3.0"
# GPU users (optional):
# uv pip install --python .venv/bin/python nvidia-cublas-cu12
```

### Download Models
```bash
# Download Piper TTS voice (~60MB)
mkdir -p voices
base=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium
curl -sL -o voices/en_US-lessac-medium.onnx "$base/en_US-lessac-medium.onnx"
curl -sL -o voices/en_US-lessac-medium.onnx.json "$base/en_US-lessac-medium.onnx.json"

# Whisper model downloads automatically on first run (~3.2GB)
```

## Repository Structure

```
opencode-voice/
├── voice-daemon.py         # Main server: loads Whisper once, serves socket IPC
├── voice-start.sh          # Key press handler: launches streamer, stops playback
├── voice-stop.sh           # Key release handler: stops stream, runs ASR, speaks/types
├── voice-cmd.sh            # Voice command dispatcher: safe app launcher
├── voice-watch.sh          # Opens OpenCode session viewer in kitty
├── voice-auto.sh           # Hands-free mode: single press + auto-endpoint
├── voice_stream.py         # Socket client: mic (parec/arecord) → PCM frames
│
├── asr.py                  # Transcriber: single model load, confidence math
├── audio_buffer.py         # Bounded deque: O(1) append, cheap tail
├── vad_gate.py             # VAD pipeline: silence detection + trim
├── normalizer.py           # Deterministic fixes: "open code" → "OpenCode"
├── config.py               # TOML config + env overrides
├── config.example.toml     # Configuration template
├── proto.py                # Socket framing: control + binary audio
├── clean_for_speech.py     # Strips code/URLs/markdown before TTS
├── transcribe.py           # One-shot VAD-gated fallback (legacy)
├── benchmark.py            # Latency/RTF reports
│
├── tests/                  # Unit tests (stdlib unittest, CUDA-free)
│   ├── test_buffer.py      # AudioBuffer tests
│   ├── test_commands.py    # Voice command parser tests
│   ├── test_confidence.py  # Confidence calculation tests
│   ├── test_normalizer.py  # Phrase normalization tests
│   ├── test_proto.py       # Socket protocol framing tests
│   └── test_server.py      # Full server integration tests
│
├── opencode-voice.service  # systemd user unit
├── hypr-voice.conf         # Bind documentation
├── LICENSE                 # MIT
├── README.md               # Full documentation
└── .gitignore
```

### Key Modules at a Glance

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `voice-daemon.py` | Server entry point | `VoiceServer`, `Session`, `load_model()` |
| `asr.py` | Transcription wrapper | `Transcriber` class |
| `audio_buffer.py` | Ring buffer for streaming | `AudioBuffer` class |
| `proto.py` | Socket IPC framing | `send_json()`, `read_frame()`, `s16_bytes_to_float32()` |
| `vad_gate.py` | Voice activity detection | `analyze()`, `analyze_array()` |
| `normalizer.py` | Phrase corrections | `normalize()`, `rules_from_config()` |

## Development Setup

### Environment Variables
Common overrides during development:
```bash
export OC_VOICE_SOCKET=/tmp/test-socket.sock    # Custom socket path
export OC_VOICE_MODEL=tiny.en                   # Smaller model for testing
export OC_VOICE_DEVICE=cpu                      # Force CPU
export OC_VOICE_DRYRUN=1                        # voice-cmd.sh dry-run
```

### Running the Server Locally
```bash
# Start the daemon in the foreground
.venv/bin/python voice-daemon.py

# Expected output:
# device: cuda (or cpu)
# listening on /run/user/1000/opencode-voice.sock
# ready
```

### Testing Without Hardware
```bash
# Run the test suite (mocked, no GPU/mic required)
.venv/bin/python -m unittest discover -s tests

# Run a specific test
.venv/bin/python -m unittest tests.test_proto
```

### Development Binds (Hyprland)
For testing, add temporary binds to `~/.config/hypr/hyprland.lua`:
```lua
-- Developer test binds (Ctrl+Alt+space instead of Super+space)
hl.bind("CTRL + ALT + space", 
  hl.dsp.exec_cmd("/path/to/opencode-voice/voice-start.sh"))
hl.bind("CTRL + ALT + space", 
  hl.dsp.exec_cmd("/path/to/opencode-voice/voice-stop.sh"), { release = true })
```

Then `hyprctl reload` and test.

## Making Changes

### File Organization
- **Daemon/server logic:** `voice-daemon.py`, `asr.py`, `audio_buffer.py`, `proto.py`, `vad_gate.py`
- **Shell scripts:** `voice-*.sh` (simple, no heavy lifting)
- **Voice commands:** `voice-cmd.sh`, `tests/test_commands.py`
- **Configuration:** `config.py`, `config.example.toml`
- **Text processing:** `normalizer.py`, `clean_for_speech.py`

### Example: Adding a Normalization Rule
1. Edit `config.example.toml`:
   ```toml
   [normalization]
   replaces = [
     ...existing...
     ["new mishearing", "Correct Form"],
   ]
   ```
2. Add a test in `tests/test_normalizer.py`
3. Run tests: `.venv/bin/python -m unittest tests.test_normalizer`

### Example: Adding a New Voice Command
1. Edit `voice-cmd.sh` in the registry section:
   ```bash
   register_app spotify "Spotify" spotify "spotify|spotify music"
   ```
2. Add test cases to `tests/test_commands.py`
3. Verify:
   ```bash
   OC_VOICE_DRYRUN=1 ./voice-cmd.sh "open spotify"
   .venv/bin/python -m unittest tests.test_commands
   ```

## Testing

### Unit Tests
```bash
# Run all tests
.venv/bin/python -m unittest discover -s tests -v

# Run a specific module
.venv/bin/python -m unittest tests.test_proto

# Run a specific test
.venv/bin/python -m unittest tests.test_buffer.AudioBufferTests.test_append
```

### Benchmark
Test latency on your hardware:
```bash
# In-process simulation (no socket, no daemon)
.venv/bin/python benchmark.py path/to/test-audio.wav

# Against a live daemon
.venv/bin/python benchmark.py --socket path/to/test-audio.wav

# Test cancellation responsiveness
.venv/bin/python benchmark.py --socket --cancel-test path/to/test-audio.wav
```

### Syntax Check
```bash
# Bash scripts
bash -n voice-*.sh

# Python
.venv/bin/python -m py_compile *.py
```

## Pull Requests

### Before You Submit
1. **Run tests:** `.venv/bin/python -m unittest discover -s tests`
2. **Syntax check:** `bash -n *.sh` and `.venv/bin/python -m py_compile *.py`
3. **Document changes:** Update README if you change user-facing behavior
4. **Add tests:** Any new logic should have corresponding tests in `tests/`
5. **Reference issue:** Link to any related issue in your PR description

### PR Title & Description
- **Title:** Concise, imperative (e.g., "Fix VAD threshold edge case", "Add confidence logging")
- **Description:**
  - What problem does it solve?
  - How does it solve it?
  - Any breaking changes?
  - Testing notes (especially if it requires hardware)

### Merge Policy
- MIT Licensed (see LICENSE)
- All tests must pass
- Code review encouraged (especially for `asr.py`, `proto.py`, `voice-daemon.py`)

## Code Style

### Python
- **Standard:** PEP 8 (enforced loosely; focus on clarity)
- **Type hints:** Appreciated but not required for all functions
- **Docstrings:** Required for public classes and functions
- **Comments:** Explain *why*, not *what*

Example:
```python
def transcribe_confident(self, audio: np.ndarray) -> tuple[str, float | None, list]:
    """Transcribe audio, return (text, confidence, segments).
    
    Confidence is duration-weighted mean of exp(segment avg_logprob),
    or None if unavailable.
    """
    ...
```

### Bash
- **Strict mode:** Use `set -euo pipefail` at the top
- **Quoting:** Quote all variables: `"$var"`, not `$var`
- **Portability:** Test on Bash 4.4+ (Hyprland standard)

Example:
```bash
#!/bin/bash
set -euo pipefail

socket_path="${OC_VOICE_SOCKET:-$XDG_RUNTIME_DIR/opencode-voice.sock}"
if ! kill -0 "$daemon_pid" 2>/dev/null; then
  echo "Daemon not running" >&2
  exit 1
fi
```

### Config (TOML)
- **Comments:** Explain non-obvious settings
- **Defaults:** Match behavior documented in README

---

## Common Tasks

### Debug Daemon Logs
```bash
# Live logs (follow mode)
journalctl --user -u opencode-voice -f

# Last 100 lines
journalctl --user -u opencode-voice -n 100
```

### Check Socket Status
```bash
# Is the daemon listening?
nc -U /run/user/1000/opencode-voice.sock &>/dev/null && echo "OK" || echo "Down"

# Files in spool (legacy mode)
ls -la /tmp/oc-voice/
```

### Add a Debugging Log Point
In `voice-daemon.py`, the `Session.log()` method writes to stdout and the journal:
```python
self.log(f"my debug info: {value}")
# Output: [voice] request=20261003-170000-abc123 my debug info: ...
```

---

## Questions?

- **Documentation:** See [README.md](README.md) — it's comprehensive
- **Code structure:** Read `voice-daemon.py` top-to-bottom; it's the entry point
- **Socket protocol:** Study `proto.py` and `tests/test_proto.py`
- **Tests:** Browse `tests/` for working examples

Good luck, and thank you for contributing!
