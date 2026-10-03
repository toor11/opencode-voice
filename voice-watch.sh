#!/bin/bash
# Open the OpenCode TUI attached to the latest voice session.
# Voice ask mode (SUPER+SHIFT+space) talks headless: this shows you the
# actual conversation. Run it directly, bind it to a key, or say
# "show session" (voice-cmd.sh).
set -u
SPOOL=/tmp/oc-voice
SESF=$SPOOL/voice.session
OPENCODE_BIN="/home/user/.opencode/bin/opencode"
# Same pinned project dir as voice-stop.sh ask mode (OC_VOICE_DIR wins).
VOICE_DIR="${OC_VOICE_DIR:-$HOME/Projects}"

SESID="$(cat "$SESF" 2>/dev/null || true)"
if [ -z "$SESID" ] && [ -d "$VOICE_DIR" ]; then
  # No voice chat yet (or /tmp was wiped): newest session in the project.
  SESID="$(cd "$VOICE_DIR" 2>/dev/null && "$OPENCODE_BIN" session list \
    --format json -n 1 2>/dev/null | jq -r '.[0].id // empty' 2>/dev/null)"
fi
if [ -z "$SESID" ]; then
  notify-send -t 4000 "Voice" "No OpenCode session yet -- ask something first" \
    2>/dev/null || true
  echo "no session" >&2
  exit 1
fi
echo "opening session $SESID"
if command -v hyprctl >/dev/null 2>&1 && hyprctl dispatch exec \
   "kitty --directory $VOICE_DIR $OPENCODE_BIN --session $SESID" 2>/dev/null; then
  exit 0
fi
nohup kitty --directory "$VOICE_DIR" "$OPENCODE_BIN" --session "$SESID" \
  >/dev/null 2>&1 &
