# opencode-voice

Push-to-talk voice control for [OpenCode](https://opencode.ai) on Hyprland/Wayland.
Hold a key, speak, release — OpenCode answers out loud. Or dictate text at the
cursor in any app (terminal, browser, documents). Everything runs **offline on
your machine**: local speech-to-text, local text-to-speech.

## Keys

| Hold | Release does |
|---|---|
| `SUPER + space` | Sends transcript to OpenCode (`run --continue`) and speaks the reply |
| `SUPER + SHIFT + space` | Types the transcript at the cursor (dictation, via `wtype`) |

Bare `space` is deliberately *not* used — it would fire on every word you type.
Holding `SUPER` again interrupts a speaking reply (barge-in).

## Architecture

```
key press   -> voice-start.sh : pw-record 16kHz mono to /tmp/oc-voice.wav,
                                stop any playing reply (barge-in)
key release -> voice-stop.sh  : stop recorder, tap-guard, transcribe, then:
  ask:  opencode run --continue -> clean_for_speech.py -> piper TTS -> pw-play
  type: wtype "<text> " into the focused window
```

Transcription is served by `voice-daemon.py`, a persistent process that loads
whisper `small.en` **once** at startup (CUDA int8, ~650MB VRAM) and answers in
~1s per utterance instead of paying a ~5s model reload on every key release.
Decoding is tuned for voice commands: `beam_size=1`, Silero VAD filter (skips
silence, kills hallucinations), English-only, plus an initial prompt seeded
with this setup's vocabulary (OpenCode, Hyprland, Garuda…).

Handoff is file-based in `/tmp/oc-voice`:

| File | Writer | Meaning |
|---|---|---|
| `req.wav` | `voice-stop.sh` (via atomic `.tmp` + rename) | new job |
| `proc.wav` | daemon (renamed from `req.wav`) | job being transcribed |
| `res.txt` | daemon (via atomic `.tmp` + rename) | transcript; `voice-stop.sh` polls up to 60s |

If the service is down or times out, `voice-stop.sh` falls back to one-shot
`transcribe.py` (same model/settings, loads on demand). Every step appends to
`/tmp/oc-voice.log` (`heard: ...`, `device: cuda/cpu`, `answer chars: ...`),
so a silent failure always leaves a trace.

Manage the daemon with `systemctl --user status opencode-voice` and
`journalctl --user -u opencode-voice` (look for `ready`).

Voice conversations accumulate in one OpenCode session (`--continue`); list it
with `opencode session list` from `~/Projects`.

## Requirements

- Garuda/Arch Linux, Hyprland (Wayland), PipeWire, `wtype`, `pw-record`/`pw-play`
- Python 3.12 venv (see install), NVIDIA GPU optional (GTX 1050 2GB tested)
- OpenCode CLI on PATH (`~/.opencode/bin/opencode`)

## Install

```bash
cd opencode-voice
uv venv .venv
uv pip install --python .venv/bin/python faster-whisper piper-tts soundfile onnxruntime "av==12.3.0" nvidia-cublas-cu12
```

(`av` is pinned: v14+ breaks faster-whisper audio decoding. `nvidia-cublas-cu12`
is the CUDA runtime for GPU transcription — CPU-only setups can skip it.)

Download the Piper voice (~60MB, ships gitignored):

```bash
mkdir -p voices
base=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium
curl -sL -o voices/en_US-lessac-medium.onnx "$base/en_US-lessac-medium.onnx"
curl -sL -o voices/en_US-lessac-medium.onnx.json "$base/en_US-lessac-medium.onnx.json"
```

Whisper models (`small.en`) download automatically on first use into the
HuggingFace cache (`~/.cache`).

Wire the keys in `~/.config/hypr/hyprland.lua` (Hyprland v0.55+ uses the Lua
config — `hyprland.conf` sources are ignored):

```lua
hl.bind(mainMod .. " + space", hl.dsp.exec_cmd("/path/to/opencode-voice/voice-start.sh"))
hl.bind(mainMod .. " + space", hl.dsp.exec_cmd("/path/to/opencode-voice/voice-stop.sh"), { release = true })
hl.bind(mainMod .. " + SHIFT + space", hl.dsp.exec_cmd("/path/to/opencode-voice/voice-start.sh"))
hl.bind(mainMod .. " + SHIFT + space", hl.dsp.exec_cmd("OC_VOICE_MODE=type /path/to/opencode-voice/voice-stop.sh"), { release = true })
```

Then `hyprctl reload` and confirm with
`hyprctl binds -j | grep space` (expect 4 entries).

## Files

| File | Purpose |
|---|---|
| `voice-start.sh` | Key-press: start mic capture, stop any playing reply |
| `voice-stop.sh` | Key-release: stop, transcribe, ask/type, speak |
| `transcribe.py` | One-shot fallback: faster-whisper `small.en`, CUDA int8 with CPU retry, beam 1, VAD filter, tech-vocabulary prompt |
| `voice-daemon.py` + `voice-daemon.sh` | Persistent model server + CUDA-env wrapper (see above) |
| `opencode-voice.service` | systemd user unit; install to `~/.config/systemd/user/`, `enable --now` |
| `clean_for_speech.py` | Strips code blocks/URLs/markdown before TTS (900-char cap) |
| `hypr-voice.conf` | Documentation of the binds (real binds live in `hyprland.lua`) |
| `voices/` | Piper voice files (gitignored, see install) |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Nothing happens at all | Binds live in `hyprland.lua`, not `hyprland.conf`. Check `hyprctl binds -j` for `space` entries |
| Empty transcript | Mic muted: `pactl set-source-mute @DEFAULT_SOURCE@ 0`. Speak 2–3s, start ~0.5s after pressing |
| `opencode: command not found` in log | Hyprland exec has a minimal PATH; script uses the absolute `$HOME/.opencode/bin/opencode` |
| CUDA lib errors | Handled: `LD_LIBRARY_PATH` is exported before Python starts (setting it inside Python does nothing), plus a CPU retry |
| VRAM pressure (2GB cards) | `small.en` holds ~650MB resident; overflow falls back to CPU automatically |
| App ignores dictation | `wtype` works in native Wayland apps only, not XWayland windows |

## License

MIT
