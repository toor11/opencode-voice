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

## How it works

```
key press   -> voice-start.sh : pw-record 16kHz mono to /tmp/oc-voice.wav
key release -> voice-stop.sh  : stop record
                               -> transcribe.py (faster-whisper, GPU if present)
                               -> ask:  opencode run --continue, piper TTS, pw-play
                                  type: wtype "<text> " into focused window
```

Steps are logged to `/tmp/oc-voice.log` (`heard: ...`, `device: cuda/cpu`,
`answer chars: ...`). Voice conversations accumulate in one OpenCode session;
list it with `opencode session list` from `~/Projects`.

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

Whisper models (`medium.en`) download automatically on first use into the
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
| `transcribe.py` | faster-whisper `medium.en`, CUDA int8 with CPU fallback, beam 5, VAD filter, tech-vocabulary prompt |
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
| VRAM pressure (2GB cards) | `medium.en` peaks ~1.5GB; overflow falls back to CPU automatically |
| App ignores dictation | `wtype` works in native Wayland apps only, not XWayland windows |

## License

MIT
