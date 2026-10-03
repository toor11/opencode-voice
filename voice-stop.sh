#!/bin/bash
# Hold release: stop capture -> transcribe (whisper tiny.en, local) ->
# ask opencode -> speak reply (piper, local).
# Called by Hyprland on key RELEASE.
set -u
export PATH="/home/user/.opencode/bin:/home/user/.local/bin:/usr/bin:/bin"
# CUDA runtime ships as pip wheels inside the venv; the loader only honors
# LD_LIBRARY_PATH when set before the python process starts.
shopt -s nullglob
DIR="$(cd "$(dirname "$0")" && pwd)"
_NVLIBS=( "$DIR"/.venv/lib/python*/site-packages/nvidia/*/lib )
shopt -u nullglob
if [ "${#_NVLIBS[@]}" -gt 0 ]; then
  _NVJOIN="$(printf '%s:' "${_NVLIBS[@]}")"
  _NVJOIN="${_NVJOIN%:}"
  export LD_LIBRARY_PATH="$_NVJOIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
WAV=/tmp/oc-voice.wav
PIDF=/tmp/oc-voice.pid
REPLY_TXT=/tmp/oc-reply.txt
REPLY_WAV=/tmp/oc-reply.wav
LOG=/tmp/oc-voice.log
OPENCODE_BIN="/home/user/.opencode/bin/opencode"
MODE="${OC_VOICE_MODE:-ask}"   # ask | type

log() { echo "$(date '+%H:%M:%S') $*" >> "$LOG"; }
log "--- release, mode=$MODE ---"

if [ -f "$PIDF" ]; then
  kill "$(cat "$PIDF")" 2>/dev/null || true
  rm -f "$PIDF"
  sleep 0.3
fi

# accidental tap guard (< ~0.4s of audio)
SIZE="$(stat -c%s "$WAV" 2>/dev/null || echo 0)"
if [ ! -f "$WAV" ] || [ "$SIZE" -lt 5000 ]; then
  log "abort: wav missing/too small (size=$SIZE)"
  rm -f "$WAV"
  exit 0
fi

SPOOL=/tmp/oc-voice
mkdir -p "$SPOOL"
TEXT=""
# Prefer the persistent daemon (model stays loaded, no per-release reload).
if systemctl --user is-active --quiet opencode-voice 2>/dev/null; then
  rm -f "$SPOOL/res.txt"
  cp "$WAV" "$SPOOL/req.tmp" && mv "$SPOOL/req.tmp" "$SPOOL/req.wav"
  for _ in $(seq 1 600); do
    [ -f "$SPOOL/res.txt" ] && break
    sleep 0.1
  done
  if [ -f "$SPOOL/res.txt" ]; then
    TEXT="$(cat "$SPOOL/res.txt")"
    log "transcribed via daemon"
  else
    log "WARN: daemon timeout, one-shot fallback"
  fi
fi
if [ -z "${TEXT// }" ]; then
  TEXT="$("$DIR/.venv/bin/python" "$DIR/transcribe.py" "$WAV" 2>>"$LOG")"
fi
rm -f "$WAV"
log "heard: $TEXT"
[ -z "${TEXT// }" ] && { log "abort: empty transcript"; exit 0; }

notify-send -t 3000 "You said" "$TEXT" 2>/dev/null || true

if [ "$MODE" = "type" ]; then
  # dictation mode: type into focused window (terminal, browser, document)
  wtype -- "$TEXT " 2>>"$LOG" || log "WARN: wtype failed"
  log "typed ${#TEXT} chars at cursor"
  exit 0
fi

# ask mode: run through opencode, keep conversing in last session
ANS="$("$OPENCODE_BIN" run --continue "$TEXT" 2>>"$LOG")"
[ -z "${ANS// }" ] && ANS="Sorry, I got no answer."
log "answer chars: ${#ANS}"
echo "$ANS" > "$REPLY_TXT"
echo "$ANS"

SPOKEN="$("$DIR/.venv/bin/python" "$DIR/clean_for_speech.py" "$REPLY_TXT" 2>>"$LOG")"
[ -z "${SPOKEN// }" ] && { log "abort: nothing speakable"; exit 0; }
echo "$SPOKEN" | "$DIR/.venv/bin/python" -m piper \
  --model "$DIR/voices/en_US-lessac-medium.onnx" \
  --output_file "$REPLY_WAV" 2>>"$LOG"
pw-play "$REPLY_WAV" 2>>"$LOG" || aplay "$REPLY_WAV" 2>>"$LOG" || log "WARN: no audio player worked"
