#!/bin/bash
# Spoken command dispatcher for opencode-voice.
# Called by voice-stop.sh with the final transcript: if it matches a
# registered command, the action runs and this exits 0 (normal ask/type
# delivery is skipped). Anything else exits 1 (not a command).
#
# SECURITY: the transcript is untrusted input and NEVER defines what gets
# executed. Parsing resolves speech to (action, registered-target) pairs;
# execution only ever runs executables from the registry below. There is no
# eval, no `sh -c`, no command substitution on transcript text anywhere here.
#
# Grammar (deterministic, full-string match only):
#   [please] (open|launch|start|run) [the] <registered app alias>
# plus a small list of exact full-phrase commands (e.g. the opencode one).
# Anything else -- "tell me about firefox", "fix firefox", "open source
# software" -- is not a command and falls through to OpenCode.
#
# Adding an app = one register_app line, nothing else. Test with:
#   OC_VOICE_DRYRUN=1 ./voice-cmd.sh "open firefox"
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENCODE_BIN="${OC_VOICE_OPENCODE:-$HOME/.opencode/bin/opencode}"
PROJECTS_DIR="$HOME/Projects"

log() { echo "$(date '+%H:%M:%S') cmd: $*" >> /tmp/oc-voice.log; }
notify() { notify-send -t 3000 "Voice command" "$1" 2>/dev/null || true; }
DRYRUN="${OC_VOICE_DRYRUN:-0}"

# ---------------- registry ----------------
# Canonical apps. aliases are |-separated, matched against the whole
# remainder of the command (exact equality after normalization).
# NOTE (Garuda): this machine's Firefox is FireDragon, so every
# firefox-shaped alias intentionally resolves to firedragon. There is no
# silent cross-browser fallback: if firedragon were missing you would get
# "not installed", never a different browser.
declare -A APP_EXE APP_DISPLAY APP_ALIASES
APP_ORDER=()
register_app() {  # canonical display exe "alias1|alias2|..."
  APP_ORDER+=("$1")
  APP_DISPLAY[$1]="$2"
  APP_EXE[$1]="$3"
  APP_ALIASES[$1]="$4"
}
register_app firedragon "FireDragon" firedragon \
  "firedragon|fire dragon|firefox|fire fox|firefox browser|mozilla|fire folks"
# ("fire folks" is what small.en hears for "firefox" on this mic; harmless
# as an alias -- no normal sentence ends "... fire folks" after open/launch.)
register_app brave "Brave" brave "brave|brave browser"
register_app chromium "Chromium" chromium "chromium|chrome|chromium browser"
register_app vscode "VS Code" code "vscode|vs code|visual studio code"
register_app opencode "OpenCode" "$OPENCODE_BIN" "opencode"

VERBS="open launch start run"

# Exact full-phrase commands: normalized string -> action.
declare -A PHRASE_ACTION
PHRASE_ACTION["open opencode and start working"]="opencode_work"
PHRASE_ACTION["show session"]="watch_session"
PHRASE_ACTION["show opencode session"]="watch_session"
PHRASE_ACTION["open session"]="watch_session"
# ("launch opencode" / "start opencode" parse via the verb+alias grammar
# and land on the same opencode action through the executor below.)

# ---------------- parsing (no side effects) ----------------
normalize_text() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9 ' ' ' \
    | tr -s ' ' | sed 's/^ *//;s/ *$//'
}

# Echoes "action|target". Target is a registry canonical name, a phrase
# action, or empty when the text is not a command.
parse_command() {
  local norm="$1" verb rest app alias
  while [[ "$norm" == "please "* ]]; do norm="${norm#please }"; done
  if [[ -n "${PHRASE_ACTION[$norm]:-}" ]]; then
    echo "${PHRASE_ACTION[$norm]}|"
    return 0
  fi
  [[ "$norm" == *" "* ]] || return 1
  # Verb is normally the first word, but small.en glues compounds
  # ("OpenFire Fox" -> "openfire fox"), so also accept the verb as a
  # prefix. Safe either way: the remainder must still equal a registered
  # alias exactly, so "opening brave" / "opencode is great" never match.
  for verb in $VERBS; do
    if [[ "$norm" == "$verb "* ]]; then
      rest="${norm#"$verb" }"
    elif [[ "$norm" == "$verb"* && "${#norm}" -gt "${#verb}" ]]; then
      rest="${norm#"$verb"}"
    else
      continue
    fi
    [[ "$rest" == "the "* ]] && rest="${rest#the }"
    for app in "${APP_ORDER[@]}"; do
      IFS='|' read -ra _aliases <<< "${APP_ALIASES[$app]}"
      for alias in "${_aliases[@]}"; do
        if [[ "$rest" == "$alias" ]]; then
          if [[ "$app" == "opencode" ]]; then
            echo "opencode_work|"
          else
            echo "launch_app|$app"
          fi
          return 0
        fi
      done
    done
  done
  return 1
}

resolve_exe() { echo "${APP_EXE[$1]:-}"; }

app_installed() {
  local exe="$1"
  if [[ "$exe" == /* ]]; then
    [[ -x "$exe" ]]
  else
    command -v "$exe" >/dev/null 2>&1
  fi
}

# ---------------- execution (registry values only) ----------------
launch_app() {  # $1 = canonical app name; exe comes from the registry
  local app="$1" exe display
  exe="$(resolve_exe "$app")"
  display="${APP_DISPLAY[$app]}"
  if [ "$DRYRUN" = "1" ]; then
    if app_installed "$exe"; then _inst=yes; else _inst=no; fi
    echo "COMMAND: launch_app"
    echo "TARGET: $app"
    echo "EXECUTABLE: $exe"
    echo "INSTALLED: $_inst"
    echo "ACTION: launch"
    return 0
  fi
  if ! app_installed "$exe"; then
    notify "$display is not installed"
    log "$display is not installed (wanted $exe)"
    return 0  # handled: never feed a failed command to OpenCode as a prompt
  fi
  notify "Opening $display"
  log "launch $app ($exe)"
  if command -v hyprctl >/dev/null 2>&1 && \
     hyprctl dispatch exec "$exe" 2>/dev/null; then
    return 0
  fi
  # Fallback: no compositor dispatch (still needs a display).
  nohup "$exe" >/dev/null 2>&1 &
}

launch_opencode_projects() {
  if [ "$DRYRUN" = "1" ]; then
    echo "COMMAND: opencode.work"
    echo "ACTION: launch_opencode_projects"
    echo "DIRECTORY: $PROJECTS_DIR"
    echo "EXECUTABLE: $OPENCODE_BIN"
    return 0
  fi
  notify "Opening opencode in ~/Projects"
  log "launch opencode in $PROJECTS_DIR"
  if ! app_installed "$OPENCODE_BIN"; then
    notify "OpenCode is not installed"
    log "opencode missing ($OPENCODE_BIN)"
    return 0
  fi
  if command -v hyprctl >/dev/null 2>&1 && hyprctl dispatch exec \
     "kitty --directory $PROJECTS_DIR $OPENCODE_BIN" 2>/dev/null; then
    return 0
  fi
  nohup kitty --directory "$PROJECTS_DIR" "$OPENCODE_BIN" >/dev/null 2>&1 &
}

watch_session() {
  # Open the TUI attached to the latest voice session (voice-watch.sh).
  if [ "$DRYRUN" = "1" ]; then
    echo "COMMAND: watch_session"
    echo "ACTION: run_voice_watch"
    echo "SCRIPT: $SCRIPT_DIR/voice-watch.sh"
    return 0
  fi
  notify "Opening voice session"
  log "watch session"
  nohup "$SCRIPT_DIR/voice-watch.sh" >/dev/null 2>&1 &
}

# ---------------- main ----------------
TEXT="${1:-}"
NORM="$(normalize_text "$TEXT")"
log "transcript=\"$TEXT\""
log "normalized=\"$NORM\""

PARSED="$(parse_command "$NORM")" || {
  log "no command match"
  exit 1
}
ACTION="${PARSED%%|*}"
TARGET="${PARSED#*|}"
log "parsed action=$ACTION target=${TARGET:-none}"

case "$ACTION" in
  launch_app)    launch_app "$TARGET" ;;
  opencode_work) launch_opencode_projects ;;
  watch_session) watch_session ;;
  *) log "unknown action: $ACTION"; exit 1 ;;
esac
exit 0
