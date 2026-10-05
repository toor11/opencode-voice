#!/bin/bash
# Hold release: finalize the streaming session -> ask opencode / type result.
# Called by Hyprland on key RELEASE.
#
# Default (socket mode): signal the voice_stream.py session (SIGUSR1 = STOP),
# wait for the daemon's result, then run the unchanged deliver pipeline.
# Legacy: OC_VOICE_LEGACY=1 keeps the old stop-capture + spool behavior.
set -u
export PATH="$HOME/.opencode/bin:$HOME/.local/bin:/usr/bin:/bin"
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

LOG=/tmp/oc-voice.log
OPENCODE_BIN="${OC_VOICE_OPENCODE:-$HOME/.opencode/bin/opencode}"
MODE="${OC_VOICE_MODE:-ask}"   # ask | type

log() { echo "$(date '+%H:%M:%S') $*" >> "$LOG"; }

if [ "${OC_VOICE_LEGACY:-0}" = "1" ]; then
  # ---------------- legacy file path (unchanged) ----------------
  WAV=/tmp/oc-voice.wav
  PIDF=/tmp/oc-voice.pid
  log "--- release, mode=$MODE (legacy) ---"
  if [ -f "$PIDF" ]; then
    kill "$(cat "$PIDF")" 2>/dev/null || true
    rm -f "$PIDF"
    sleep 0.3
  fi
  pkill -f "pw-record.*oc-voice[.]wav" 2>/dev/null || true
  SIZE="$(stat -c%s "$WAV" 2>/dev/null || echo 0)"
  if [ ! -f "$WAV" ] || [ "$SIZE" -lt 5000 ]; then
    log "abort: wav missing/too small (size=$SIZE)"
    rm -f "$WAV"
    exit 0
  fi
  MAXSIZE=3000000
  if [ "$SIZE" -gt "$MAXSIZE" ]; then
    log "abort: wav too large (size=$SIZE ~$((SIZE / 31000))s), likely missed key-release; discarded"
    notify-send -t 4000 "Voice: recording too long (~$((SIZE / 31000))s)" "Missed key-release? Discarded." 2>/dev/null || true
    rm -f "$WAV"
    exit 0
  fi
  SPOOL=/tmp/oc-voice
  mkdir -p "$SPOOL"
  TEXT=""
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
  cp -f "$WAV" /tmp/oc-voice-last.wav 2>/dev/null || true
  rm -f "$WAV"
else
  # ---------------- socket path ----------------
  SPOOL=/tmp/oc-voice
  PIDF=$SPOOL/stream.pid
  RESF=$SPOOL/stream.res
  CONFF=$SPOOL/stream.conf
  ERRF=$SPOOL/stream.err
  log "--- release, mode=$MODE (socket) ---"
  if [ -f "$PIDF" ]; then
    kill -USR1 "$(cat "$PIDF")" 2>/dev/null || true  # STOP: finalize
  else
    log "note: no active streamer (auto mode may have finished already)"
  fi
  TEXT=""
  CONF=""
  for _ in $(seq 1 600); do  # up to 60s for the final transcript
    [ -f "$RESF" ] && { TEXT="$(cat "$RESF")"; break; }
    [ -f "$ERRF" ] && break
    sleep 0.1
  done
  [ -f "$CONFF" ] && CONF="$(cat "$CONFF")"
  case "$CONF" in
    *low) LOWCONF=1 ;;
    *) LOWCONF=0 ;;
  esac
  if [ -f "$ERRF" ] && [ -z "${TEXT// }" ]; then
    ERR="$(cat "$ERRF")"
    log "abort: streamer error: $ERR"
    notify-send -t 4000 "Voice error" "$ERR" 2>/dev/null || true
    rm -f "$RESF" "$ERRF"
    exit 1
  fi
  if [ ! -f "$RESF" ]; then
    log "abort: no result (streamer died? daemon down?)"
    notify-send -t 4000 "Voice: no result" "Daemon not running?" 2>/dev/null || true
    exit 1
  fi
  rm -f "$RESF" "$ERRF" "$CONFF"
fi

# ---------------- spoken commands (shared) ----------------
# Phrases like "open opencode and start working" run an action instead of
# normal delivery (works in both ask and type modes).
if [ -n "${TEXT// }" ] && "$DIR/voice-cmd.sh" "$TEXT"; then
  log "handled as voice command"
  exit 0
fi

# ---------------- deliver (shared) ----------------
log "heard: $TEXT (conf: ${CONF:-n/a})"
[ -z "${TEXT// }" ] && { log "abort: empty transcript"; exit 0; }

if [ "${LOWCONF:-0}" = "1" ]; then
  notify-send -t 3000 "You said (unsure)" "$TEXT" 2>/dev/null || true
else
  notify-send -t 3000 "You said" "$TEXT" 2>/dev/null || true
fi

if [ "$MODE" = "type" ]; then
  # dictation mode: type into focused window (terminal, browser, document)
  wtype -- "$TEXT " 2>>"$LOG" || log "WARN: wtype failed"
  log "typed ${#TEXT} chars at cursor"
  exit 0
fi

# ask mode: run through opencode, keep conversing in last session.
# Voice chats live in one pinned project directory (default ~/Projects):
# Hyprland launches this script with $HOME as CWD, which would scatter
# chats into the invisible "global" project whose sessions even
# `session list` cannot see. Pinning keeps ask + tracking + watch agreed.
REPLY_TXT=/tmp/oc-reply.txt
REPLY_WAV=/tmp/oc-reply.wav
VOICE_DIR="${OC_VOICE_DIR:-$HOME/Projects}"
if [ -d "$VOICE_DIR" ]; then
  cd "$VOICE_DIR" || log "WARN: cannot cd to $VOICE_DIR"
else
  log "WARN: voice dir missing: $VOICE_DIR (using $PWD)"
  VOICE_DIR="$PWD"
fi
ANS="$("$OPENCODE_BIN" run --continue "$TEXT" 2>>"$LOG")"
[ -z "${ANS// }" ] && ANS="Sorry, I got no answer."
log "answer chars: ${#ANS}"
echo "$ANS" > "$REPLY_TXT"
echo "$ANS"
# Remember which session this went to, so voice-watch.sh (or the
# "show session" voice command) can open the TUI attached to it.
SESF=$SPOOL/voice.session
SESID="$("$OPENCODE_BIN" session list \
  --format json -n 1 2>/dev/null | jq -r '.[0].id // empty' 2>/dev/null)"
if [ -n "$SESID" ]; then
  echo "$SESID" > "$SESF"
  log "session: $SESID"
fi

SPOKEN="$("$DIR/.venv/bin/python" "$DIR/clean_for_speech.py" "$REPLY_TXT" 2>>"$LOG")"
[ -z "${SPOKEN// }" ] && { log "abort: nothing speakable"; exit 0; }
echo "$SPOKEN" | "$DIR/.venv/bin/python" -m piper \
  --model "$DIR/voices/en_US-lessac-medium.onnx" \
  --output_file "$REPLY_WAV" 2>>"$LOG"
pw-play "$REPLY_WAV" 2>>"$LOG" || aplay "$REPLY_WAV" 2>>"$LOG" || log "WARN: no audio player worked"
