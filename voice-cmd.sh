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

# "open firefox" / "launch brave" / ...: first installed match wins.
open_browser() {
  local want="$1" prog
  for prog in $want brave chromium firefox zen-browser librewolf google-chrome; do
    if command -v "$prog" >/dev/null 2>&1; then
      if [ "${OC_VOICE_DRYRUN:-0}" = "1" ]; then
        echo "ACTION: launch $prog"
        return 0
      fi
      notify "Opening $prog"
      log "opening $prog"
      if command -v hyprctl >/dev/null 2>&1; then
        hyprctl dispatch exec "$prog" 2>/dev/null && return 0
      fi
      nohup "$prog" >/dev/null 2>&1 &
      return 0
    fi
  done
  notify "No browser found ($want)"
  log "no browser found for '$want'"
  return 0  # handled (don't feed it to opencode as a question)
}

case "$NORM" in
  *opencode*work*|*opencode*start*|*launch*opencode*|\
  *open\ code*work*|*open\ code*start*)
    # "open opencode and start working", "launch opencode", ...
    open_opencode_projects
    exit 0
    ;;
  *open*firefox*|*launch*firefox*|*start*firefox*)
    # "open firefox" (falls back to any installed browser)
    open_browser firefox
    exit 0
    ;;
  *open*brave*|*launch*brave*|*start*brave*)
    open_browser brave
    exit 0
    ;;
  *open*chromium*|*launch*chromium*|*start*chromium*)
    open_browser chromium
    exit 0
    ;;
esac

exit 1
