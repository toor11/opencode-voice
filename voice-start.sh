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

# ignore key-repeat while already recording
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  exit 0
fi

rm -f "$WAV"
pw-record --rate 16000 --channels 1 "$WAV" &
echo $! > "$PIDF"
notify-send -t 1500 "listening…" "Release to send" 2>/dev/null || true
