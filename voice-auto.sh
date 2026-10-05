#!/bin/bash
# Voice-activity mode: SINGLE press, speak, auto-submit on trailing silence.
#
# Starts a streamer with VAD endpointing (OC_VOICE_AUTO=1), waits until it
# submits itself (or a timeout hits), then runs the normal release pipeline.
# No key release needed. Same OC_VOICE_MODE (ask|type) and tunables:
# END_SILENCE is fixed in voice_stream.py (--end-silence 1.2 default);
# NO_SPEECH_TIMEOUT / MAX_HOLD via env below.
set -u
export PATH="$HOME/.opencode/bin:$HOME/.local/bin:/usr/bin:/bin"
DIR="$(cd "$(dirname "$0")" && pwd)"
SPOOL=/tmp/oc-voice
LOG=/tmp/oc-voice.log
MAX_HOLD="${MAX_HOLD:-45}"

export OC_VOICE_AUTO=1
export OC_VOICE_MODE="${OC_VOICE_MODE:-ask}"

# In legacy mode there is no streamer: fall back to the wav-based auto loop.
if [ "${OC_VOICE_LEGACY:-0}" = "1" ]; then
  echo "$(date '+%H:%M:%S') auto: legacy mode not supported, use socket mode" >> "$LOG"
  notify-send -t 3000 "Voice auto" "Legacy mode has no hands-free support" 2>/dev/null || true
  exit 1
fi

"$DIR/voice-start.sh" || exit 1

# Wait for the streamer to endpoint itself (writes stream.res / stream.err).
for _ in $(seq 1 $((MAX_HOLD * 2))); do
  [ -f "$SPOOL/stream.res" ] && break
  [ -f "$SPOOL/stream.err" ] && break
  # Streamer crashed without writing either file: stop waiting.
  if [ ! -f "$SPOOL/stream.pid" ] || ! kill -0 "$(cat "$SPOOL/stream.pid" 2>/dev/null || echo)" 2>/dev/null; then
    sleep 0.5  # one grace beat: result may land just as the pid exits
    { [ -f "$SPOOL/stream.res" ] || [ -f "$SPOOL/stream.err" ]; } && break
    echo "$(date '+%H:%M:%S') auto: streamer died with no result" >> "$LOG"
    exit 1
  fi
  sleep 0.5
done

# Hand off: the streamer already finalized (or the wait timed out, in which
# case voice-stop.sh's SIGUSR1 finalizes it). Delivers ask/type as usual.
exec "$DIR/voice-stop.sh"
