#!/bin/bash
# Spoken command dispatcher for opencode-voice.
# Called by voice-stop.sh with the final transcript: if it matches a known
# command phrase, the action runs and this exits 0 (normal ask/type delivery
# is skipped). Anything else exits 1 (not a command, deliver normally).
#
# Matching is fuzzy on purpose (ASR never returns the exact words): the text
# is lowercased, punctuation stripped, then matched against glob patterns.
# Add new commands as extra case branches. Test with:
#   OC_VOICE_DRYRUN=1 ./voice-cmd.sh "open opencode and start working"
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
TEXT="${1:-}"
OPENCODE_BIN="/home/user/.opencode/bin/opencode"

log() { echo "$(date '+%H:%M:%S') cmd: $*" >> /tmp/oc-voice.log; }
notify() { notify-send -t 3000 "Voice command" "$1" 2>/dev/null || true; }

NORM="$(echo "$TEXT" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9 ' ' ')"

open_opencode_projects() {
  local dir="$HOME/Projects"
  if [ "${OC_VOICE_DRYRUN:-0}" = "1" ]; then
    echo "ACTION: open kitty in $dir running opencode"
    return 0
  fi
  notify "Opening opencode in ~/Projects"
  log "opening opencode in $dir"
  if command -v hyprctl >/dev/null 2>&1 && \
     hyprctl dispatch exec "kitty --directory $dir $OPENCODE_BIN" 2>/dev/null; then
    return 0
  fi
  # Fallback: no compositor dispatch (still needs a display).
  nohup kitty --directory "$dir" "$OPENCODE_BIN" >/dev/null 2>&1 &
}

case "$NORM" in
  *opencode*work*|*opencode*start*|*launch*opencode*|\
  *open\ code*work*|*open\ code*start*)
    # "open opencode and start working", "launch opencode", ...
    open_opencode_projects
    exit 0
    ;;
esac

exit 1
