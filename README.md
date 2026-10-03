# opencode-voice

Push-to-talk voice control for [OpenCode](https://opencode.ai) on Hyprland/Wayland.
Hold a key, speak, release — OpenCode answers out loud. Or dictate text at the
cursor in any app (terminal, browser, documents). Everything runs **offline on
your machine**: local speech-to-text, local text-to-speech.

## Keys

| Hold | Release does |
|---|---|
| `SUPER + space` | Types the transcript at the cursor (dictation, via `wtype`) |
| `SUPER + SHIFT + space` | Sends transcript to OpenCode (`run --continue`) and speaks the reply |

Hands-free alternative: bind a key to `voice-auto.sh` — single press, speak,
pause (~1.2s silence) to auto-submit, no release needed. Same `OC_VOICE_MODE`
(`ask` default, `type` for dictation). The streamer also accepts `--auto`
directly (`voice_stream.py --auto`: local VAD endpointing, SIGUSR1 still
works as a manual submit).

Bare `space` is deliberately *not* used — it would fire on every word you type.
Holding `SUPER` again interrupts a speaking reply (barge-in).

## Architecture

```
key press   -> voice-start.sh : launch voice_stream.py (mic -> socket),
                                stop any playing reply (barge-in)
key release -> voice-stop.sh  : SIGUSR1 = STOP, wait for result, then:
  ask:  opencode run --continue -> clean_for_speech.py -> piper TTS -> pw-play
  type: wtype "<text> " into the focused window
```

`voice-daemon.py` is a persistent process: whisper `small.en` loads **once**
at startup (CUDA int8, ~650MB VRAM — fits the GTX 1050 2GB) and every request
reuses that single instance. No per-utterance reload, no second model.

```
              Hyprland hotkey
                    │
                    ▼
             voice-start/client
                    │ Unix domain socket (100ms PCM frames, never blocked by ASR)
                    ▼
             ┌──────────────┐
             │ voice-daemon │
             │              │
Microphone ─►│ audio buffer │  bounded deque, O(1) append, cheap tail
             │      │       │
             │      ▼       │
             │     VAD      │  silence skipped, speech trimmed
             │      │       │
             │      ▼       │
             │ scheduled    │  FINAL_ONLY default: ONE inference at STOP
             │   ASR        │  optional LOW_COST_PARTIAL preview window
             │      │       │
             │      ▼       │
             │ confidence + │  exp(avg_logprob), then deterministic fixes
             │ normalizer   │
             └──────┬───────┘
                    │ final transcript (authoritative)
                    ▼
              OpenCode / wtype
```

VAD runs **before** Whisper (`vad_gate.py`, Silero `silero_vad_v6.onnx` on
CPU — bundled with faster-whisper, no new dependency). Silence returns empty
without waking the GPU (~0.14s, zero ASR calls); mostly-silence buffers are
trimmed in-memory first. Decoding keeps `beam_size=1`, `vad_filter=True` as
second-stage safety, English-only, plus an initial prompt seeded with this
setup's vocabulary (OpenCode, Hyprland, Garuda…).

### Streaming: audio streams, ASR is scheduled (chunked, honest)

Three distinct things, kept distinct: **audio streaming** (mic→socket→buffer
is genuinely incremental and never blocks on ASR), **incremental ASR**
(optional bounded-window previews), **token streaming** (not available —
faster-whisper 1.2.1/CTranslate2 4.8.2 `transcribe()` is array-at-once with
no interruption API; there is no token callback to use).

Default is **FINAL_ONLY**: the daemon buffers audio and runs exactly ONE
inference at STOP — no GPU work while you speak. This is the fastest mode on
the GTX 1050 and the right default for 2GB VRAM.

**LOW_COST_PARTIAL** (`[streaming] enabled = true`) adds preview text: a
small 4s window every 3s once ≥2s of *new* speech arrived, skipped whenever
the model is busy (at most one pending partial, coalesced — overlapping
inference is impossible by construction: one worker, one model lock, one
`busy` flag). Partials never delay the final: STOP discards pending preview
work and prioritizes the single authoritative transcribe. A `partial`
**replaces** the previous one; only `result` counts. Partials fire only
while speech is ongoing (recent 1.5s), so pauses don't flicker prompt echoes.

Measured on GTX 1050 / `small.en` int8 (`python benchmark.py --compare`,
same 3.67s clip): FINAL_ONLY = 1 inference / 0.81s compute; PARTIAL = 2
inferences / 0.93s compute; identical final text and confidence. Partials
cost GPU work and buy only preview text — hence off by default.

The audio buffer is a bounded deque (120s cap): O(1) append, tail extraction
copies only the window, one full copy at STOP. The old
`np.concatenate(all_chunks)`-per-partial pattern is gone.

### Confidence (a signal, not a grade)

Every result carries `confidence`: the duration-weighted mean of
`exp(segment avg_logprob)` from faster-whisper (e.g. partial 0.72, final 0.75
on the reference clip). Segments without probability data yield `null`, never
a fabricated number. Below `low_confidence_threshold` (0.55) the text is
still returned and typed — flagged (`low_confidence: true`, "You said
(unsure)" notification, `conf: … low` in the log), never silently dropped and
never "corrected" by an LLM.

### Normalization (deterministic, no LLM)

After ASR, before OpenCode: ordered word-boundary phrase fixes for known
misrecognitions (`open code`→`OpenCode`, `get hub`→`GitHub`, `cue wen`→`Qwen`,
…). Multi-word defaults only, so ordinary English (`rust is fast`, `code`,
`docker run hello`) passes through untouched. Configure in
`[normalization]` (`enabled`, `replaces` — a user list replaces defaults);
`OC_VOICE_NORMALIZE=0` disables.

### Cancellation (two levels, honest)

The session runs a socket reader plus an ASR worker, guarded by a generation
counter. **Level 1 — logical (always works):** the reader answers `CANCEL`
immediately (measured 1ms) while the worker abandons in-flight work; every
ASR job captures its generation and output from a stale generation is
discarded before any send — a cancelled request can never emit a late result,
and request B can't hear request A. **Level 2 — compute: not available.**
faster-whisper/CTranslate2 expose no interruption API (`generate()` is a
blocking C++ call; a Python Event cannot preempt a CUDA kernel), so an
in-flight inference runs to completion and is discarded — no new work starts.
No worker-process isolation either, deliberately: reloading `small.en` costs
~2.2s and risks dual VRAM residency on the 2GB card for no latency win
(the reply itself is already instant).

Pressing the push-to-talk key during playback still uses barge-in; a raw
`kill -TERM <streamer-pid>` cancels a recording (`cancelled by daemon` path).
`[cancellation] enabled = false` makes CANCEL behave like STOP.

### Socket IPC (no polling, no WAV)

Default socket: `$XDG_RUNTIME_DIR/opencode-voice.sock`, fallback
`/tmp/oc-voice/voice.sock` (`OC_VOICE_SOCKET` or `[server] socket` overrides).
Framing is `kind(1B) + len(4B BE) + payload`: kind `0x01` = JSON control,
kind `0x02` = raw s16le mono 16kHz PCM (no base64). One recording session =
one connection = one `request_id` (`20261003-170000-abc123de`); a second
concurrent session gets `{"type":"busy"}` and is disconnected — except the
client retries `start` a few times first, so a rapid re-press right after a
result lands connects instead of failing. Messages:
`start` → `started`, binary `audio…`, `stop` → `partial…` → `result`
(`text` + `timings`), `cancel` → `cancelled`, plus `error`. Length-delimited
frames tolerate partial reads and coalescing; disconnects free the busy slot
so the next press works.

`/tmp/oc-voice` now only holds tiny rendezvous files (`stream.pid`,
`stream.req`, `stream.res`/`stream.conf`/`stream.err`); no `req.wav`/`res.txt`
in normal operation. Every request logs `[voice] request=… started /
first_partial=…s confidence=… / stop / final confidence=… norm=…
final_latency=…s total=…s` to the journal and `/tmp/oc-voice.log`.

Legacy WAV spool (`req.wav` → `proc.wav` → `res.txt` polling) is kept as a
fallback: `OC_VOICE_LEGACY=1` on the scripts + `--legacy-file-mode`
(or `[server] legacy_file_mode = true`) on the daemon.

Manage the daemon with `systemctl --user status opencode-voice` and
`journalctl --user -u opencode-voice` (look for `ready`).

Voice conversations accumulate in one OpenCode session (`--continue`); list it
with `opencode session list` from `~/Projects`.

## Requirements

- Garuda/Arch Linux, Hyprland (Wayland), PipeWire, `wtype`, `parec` (mic
  streaming; `arecord` fallback), `pw-play`
- Python 3.12 venv (see install), NVIDIA GPU optional (GTX 1050 2GB tested)
- OpenCode CLI on PATH (`~/.opencode/bin/opencode`)
- No new Python dependencies: socket client/config/tests use stdlib + numpy
  (stdlib `tomllib` reads the config; Python ≥3.11 required for that)

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
| `voice-start.sh` | Key-press: launch streamer, stop any playing reply |
| `voice-stop.sh` | Key-release: STOP the stream, wait for result, ask/type, speak (legacy branch kept under `OC_VOICE_LEGACY=1`) |
| `voice-cmd.sh` | Spoken command dispatcher: transcript → safe registry parser → whitelisted app launch (exit 0 = handled, 1 = normal delivery) |
| `voice-watch.sh` | Opens kitty with the OpenCode TUI attached to the latest voice session (`opencode --session`, also via "show session") |
| `voice-auto.sh` | Hands-free: single press, streamer self-endpoints on silence, then the normal release path |
| `voice_stream.py` | Socket client: mic (`parec`→`arecord`) → PCM frames; SIGUSR1=STOP, SIGTERM=cancel, `--auto` endpoints locally |
| `voice-daemon.py` + `voice-daemon.sh` | Persistent model server: socket IPC + chunked ASR + VAD gate + confidence + normalization + cancellation (`--legacy-file-mode` re-enables the WAV spool) |
| `asr.py` | Shared single-model `Transcriber` + honest confidence math |
| `audio_buffer.py` | Bounded deque audio buffer (O(1) append, cheap tail, one copy at STOP) |
| `normalizer.py` | Deterministic phrase fixes (word-boundary, ordered, disableable) |
| `proto.py` | Socket framing + control messages (shared by daemon, client, tests, benchmark) |
| `config.py` + `config.example.toml` | TOML config + env overrides + socket resolution |
| `vad_gate.py` | Pipeline VAD: speech/silence gate + speech-only trim (file CLI + in-memory API) |
| `transcribe.py` | One-shot VAD-gated fallback used by legacy mode |
| `benchmark.py` | Latency/RTF/confidence reports: in-process, live-daemon (`--socket`), cancel reaction (`--socket --cancel-test`) |
| `tests/` | `unittest` suite, CUDA-free (mocked model): `python -m unittest discover -s tests` |
| `opencode-voice.service` | systemd user unit; install to `~/.config/systemd/user/`, `enable --now` |
| `clean_for_speech.py` | Strips code blocks/URLs/markdown before TTS (900-char cap) |
| `hypr-voice.conf` | Documentation of the binds (real binds live in `hyprland.lua`) |
| `voices/` | Piper voice files (gitignored, see install) |

## Voice commands

Certain phrases run an action instead of being sent to OpenCode. Say the
app name with any of **open / launch / start / run** (optional `please`
or `the` also match):

| Say | Does |
|---|---|
| Open FireDragon | Opens FireDragon (also matches "open firefox" -- on Garuda, Firefox *is* FireDragon, including ASR variants like "OpenFire Fox") |
| Launch Brave | Opens Brave |
| Start Chromium | Opens Chromium |
| Open VS Code | Opens VS Code (`code`; reports "not installed" if absent) |
| Open OpenCode and start working | Opens kitty in `~/Projects` running opencode (also "launch/start opencode") |

Anything else -- `"tell me about firefox"`, `"fix firefox"`,
`"open the Firefox configuration file"` -- is not a command and goes to
OpenCode normally. If a registered app isn't installed you get a
"not installed" notification instead of a different app opening.

### Add your own commands

One line in `voice-cmd.sh`, in the registry section:

```bash
register_app spotify "Spotify" spotify "spotify|spotify music"
```

The four fields are: canonical name, display name (for notifications),
executable, and `|`-separated aliases. Then these work automatically:

```text
open spotify
launch spotify music
please start spotify
```

Tips:

- **Aliases cover ASR mishearings.** Whisper rarely returns brand names
  cleanly ("firefox" arrives as "OpenFire Fox"), so add every variant
  you see in `/tmp/oc-voice.log` (`heard:` lines) as an alias.
- **Keep aliases specific.** The remainder after the verb must equal an
  alias exactly, so `"open code"` never triggers anything and normal
  dictation can't misfire -- but don't register bare words like `code`.
- **Special actions** (like opening opencode inside kitty in a fixed
  directory) need a small function plus routing in the executor, following
  the existing `launch_opencode_projects` example.

Verify before pushing:

```bash
OC_VOICE_DRYRUN=1 ./voice-cmd.sh "open spotify"   # shows COMMAND/TARGET/EXECUTABLE, launches nothing
bash -n voice-cmd.sh                               # syntax check
.venv/bin/python -m unittest tests.test_commands   # parser suite: add your phrases here
```

### Chatbot mode with a visible session

`SUPER+SHIFT+space` (ask mode) talks to OpenCode headless and speaks the
reply -- the conversation continues the latest session pinned to
`~/Projects` (`OC_VOICE_DIR` overrides). Pinning matters: without it the
chat would land in the catch-all `global` project whose sessions even
`opencode session list` cannot see.
To **see** it, run `voice-watch.sh` (bind it to a key, e.g.
`SUPER+ALT+space`), or just say **"show session"** on your next
recording. It opens kitty with the OpenCode TUI attached to your latest
voice chat (`opencode --session <id>`), so you can read history, scroll,
and type follow-ups. Every ask also records its session id to
`/tmp/oc-voice/voice.session` (plus the daemon log) for scripting.

### Security note

The transcript never defines what executes: speech resolves to a
registered target and only the registered executable runs (checked with
`command -v` first). Arbitrary shell commands are intentionally
unsupported -- no `eval`, no `sh -c`, no substitution of transcript text.

## Configuration

Optional. Copy `config.example.toml` to `~/.config/opencode-voice/config.toml`
and tune: socket path, chunk size, model/device, VAD (`threshold`,
`min_silence_duration_ms`, `speech_pad_ms`, … — pauses between words survive
by default), streaming mode (`enabled`, `interval_ms`, `window_s`,
`min_new_speech_s` — FINAL_ONLY by default), confidence
(`confidence_enabled`, `low_confidence_threshold`), normalization vocabulary
(`enabled`, `replaces`), cancellation. Every key is also
settable via `OC_VOICE_*` env vars (see `config.py`). Defaults match the
previous behavior, so no config is needed for a standard install.

## Performance

```bash
python benchmark.py /tmp/oc-voice-last.wav            # in-process simulation
python benchmark.py --socket /tmp/oc-voice-last.wav   # through the live daemon
python benchmark.py --socket --cancel-test clip.wav   # CANCEL responsiveness
python benchmark.py --compare clip.wav                # FINAL_ONLY vs PARTIALS
```

Reports audio duration, partial count, first-partial latency, final latency
(time from audio end → transcript), compute total, RTF, and average/final
confidence. The number that matters for feel is **final latency** — currently
~0.5s on the GTX 1050. Daemon logs use the same vocabulary per request:
`started`, `first_partial=…s confidence=…`, `stop`, `final
confidence=… final_latency=…s total=…s` (transcript text itself is never
logged server-side).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Nothing happens at all | Binds live in `hyprland.lua`, not `hyprland.conf`. Check `hyprctl binds -j` for `space` entries |
| `Voice error: daemon unavailable` | Daemon down: `systemctl --user status opencode-voice`, `journalctl --user -u opencode-voice` (look for `ready`). Socket lives at `$XDG_RUNTIME_DIR/opencode-voice.sock`, fallback `/tmp/oc-voice/voice.sock` |
| Socket permission errors | Socket dir must be user-writable; `OC_VOICE_SOCKET` can point elsewhere. Never TCP — no ports involved |
| `microphone unavailable` in log | Neither `parec` (pipewire-pulse) nor `arecord` could open the mic; `pactl set-source-mute @DEFAULT_SOURCE@ 0`, check `pactl info` |
| CUDA lib errors | `LD_LIBRARY_PATH` is exported before Python starts (daemon wrapper, stop script, benchmark re-exec); plus CPU fallback |
| CUDA unavailable at runtime | Daemon logs `cuda failed…, using cpu` and keeps serving — slower, same protocol |
| ASR error mid-session (e.g. CUDA failure) | Session returns an `error` message instead of hanging; the daemon stays up and the next press works |
| VRAM pressure (2GB cards) | One resident `small.en` int8 (~650MB, ~820MiB total with CUDA workspace, flat across sessions); partial windows are bounded (4s) so cost stays flat |
| Empty transcript | Silence-only rooms return empty by design (no GPU wake). Otherwise: mic muted, or speak 2–3s starting ~0.5s after pressing |
| "You said (unsure)" notification | Confidence below threshold (`low_confidence_threshold`, 0.55) — text still typed, just flagged. Tune or disable via `[transcription]` / `OC_VOICE_LOW_CONF` |
| `opencode: command not found` in log | Hyprland exec has a minimal PATH; script uses the absolute `$HOME/.opencode/bin/opencode` |
| App ignores dictation | `wtype` works in native Wayland apps only, not XWayland windows |

## License

MIT
