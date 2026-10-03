#!/bin/bash
# Hold start: begin a streaming session, interrupt any reply currently playing.
# Called by Hyprland on key PRESS.
#
# Default (socket mode): launches voice_stream.py, which captures the mic
# (parec/arecord) and streams PCM to the daemon over the Unix socket.
# Legacy: OC_VOICE_LEGACY=1 keeps the old pw-record-to-wav behavior.
set -u
export PATH="/home/user/.local/bin:/usr/bin:/bin"
DIR="$(cd "$(dirname "$0")" && pwd)"
# Mic can come back muted after a reboot; ensure it is live on every press.
pactl set-source-mute @DEFAULT_SOURCE@ 0 2>/dev/null || true

# barge-in: stop previous speech
pkill -f "pw-play.*oc-reply" 2>/dev/null || true

if [ "${OC_VOICE_LEGACY:-0}" = "1" ]; then
  WAV=/tmp/oc-voice.wav
  PIDF=/tmp/oc-voice.pid
  MAX_HOLD=75
  if [ -f "$PIDF" ]; then
    OLDPID="$(cat "$PIDF" 2>/dev/null || true)"
    if [ -n "${OLDPID:-}" ] && kill -0 "$OLDPID" 2>/dev/null; then
      NOW="$(date +%s)"
      BORN="$(stat -c %Y "$PIDF" 2>/dev/null || echo "$NOW")"
      if [ "$((NOW - BORN))" -lt "$MAX_HOLD" ]; then
        exit 0
      fi
      kill "$OLDPID" 2>/dev/null || true
      sleep 0.3
      pkill -f "pw-record.*oc-voice[.]wav" 2>/dev/null || true
    fi
  fi
  rm -f "$WAV" "$PIDF"
  timeout 60 pw-record --rate 16000 --channels 1 "$WAV" &
  echo $! > "$PIDF"
  notify-send -t 1500 "listening…" "Release to send" 2>/dev/null || true
  exit 0
fi

# --- socket mode ---
SPOOL=/tmp/oc-voice
mkdir -p "$SPOOL"
PIDF=$SPOOL/stream.pid
MAX_HOLD=75  # reap orphaned streamers (a missed key-release leaves one running)
if [ -f "$PIDF" ]; then
  OLDPID="$(cat "$PIDF" 2>/dev/null || true)"
  if [ -n "${OLDPID:-}" ] && kill -0 "$OLDPID" 2>/dev/null; then
    NOW="$(date +%s)"
    BORN="$(stat -c %Y "$PIDF" 2>/dev/null || echo "$NOW")"
    if [ "$((NOW - BORN))" -lt "$MAX_HOLD" ]; then
      exit 0  # key-repeat while legitimately recording
    fi
    kill "$OLDPID" 2>/dev/null || true  # stale: missed release; reap it
    sleep 0.5
    kill -9 "$OLDPID" 2>/dev/null || true
  fi
fi

rm -f "$SPOOL/stream.res" "$SPOOL/stream.err"
AUTO=()
if [ "${OC_VOICE_AUTO:-0}" = "1" ]; then
  AUTO=(--auto)
fi
# shellcheck disable=SC2086
nohup "$DIR/.venv/bin/python" "$DIR/voice_stream.py" ${AUTO[@]+"${AUTO[@]}"} \
  >>/tmp/oc-voice.log 2>&1 &
# The venv interpreter needs ~1s for imports before the pidfile appears;
# poll (don't snapshot at 0.3s -- that false-fails and orphans a recorder).
READY=0
for _ in $(seq 1 40); do
  if [ -f "$SPOOL/stream.err" ]; then
    break
  fi
  if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    READY=1
    break
  fi
  sleep 0.1
done
if [ "$READY" = "1" ]; then
  if [ "${OC_VOICE_AUTO:-0}" = "1" ]; then
    notify-send -t 2000 "listening (auto)…" "Pause to send" 2>/dev/null || true
  else
    notify-send -t 1500 "listening…" "Release to send" 2>/dev/null || true
  fi
else
  ERR="$(cat "$SPOOL/stream.err" 2>/dev/null || echo 'streamer failed to start')"
  notify-send -t 4000 "Voice: $ERR" "Is the daemon running?" 2>/dev/null || true
  echo "$(date '+%H:%M:%S') start: $ERR" >> /tmp/oc-voice.log
  exit 1
fi
