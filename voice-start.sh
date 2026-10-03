#!/bin/bash
# Hold start: begin mic capture, interrupt any reply currently playing.
# Called by Hyprland on key PRESS.
set -u
export PATH="/home/user/.local/bin:/usr/bin:/bin"
# Mic can come back muted after a reboot; ensure it is live on every press.
pactl set-source-mute @DEFAULT_SOURCE@ 0 2>/dev/null || true
WAV=/tmp/oc-voice.wav
PIDF=/tmp/oc-voice.pid

# barge-in: stop previous speech
pkill -f "pw-play.*oc-reply" 2>/dev/null

# key-repeat guard: ignore repeats while a fresh recording is in progress,
# but reap orphaned recorders (a missed key-release leaves pw-record running).
MAX_HOLD=75  # seconds: recordings older than this mean the release was missed
if [ -f "$PIDF" ]; then
  OLDPID="$(cat "$PIDF" 2>/dev/null || true)"
  if [ -n "${OLDPID:-}" ] && kill -0 "$OLDPID" 2>/dev/null; then
    NOW="$(date +%s)"
    BORN="$(stat -c %Y "$PIDF" 2>/dev/null || echo "$NOW")"
    if [ "$((NOW - BORN))" -lt "$MAX_HOLD" ]; then
      exit 0  # key-repeat while legitimately recording
    fi
    # stale: a missed key-release left the recorder running; reap it
    # (OLDPID is the `timeout` wrapper, so also catch its pw-record child)
    kill "$OLDPID" 2>/dev/null || true
    sleep 0.3
    pkill -f "pw-record.*oc-voice[.]wav" 2>/dev/null || true
  fi
fi

rm -f "$WAV" "$PIDF"
# timeout caps the capture even if the release event never arrives
timeout 60 pw-record --rate 16000 --channels 1 "$WAV" &
echo $! > "$PIDF"
notify-send -t 1500 "listening…" "Release to send" 2>/dev/null || true
