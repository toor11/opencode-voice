#!/bin/bash
# Wrapper: export CUDA runtime path BEFORE python starts (the loader ignores
# LD_LIBRARY_PATH set from inside the process), then exec the daemon.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
shopt -s nullglob
_NVLIBS=( "$DIR"/.venv/lib/python*/site-packages/nvidia/*/lib )
shopt -u nullglob
if [ "${#_NVLIBS[@]}" -gt 0 ]; then
  _NVJOIN="$(printf '%s:' "${_NVLIBS[@]}")"
  _NVJOIN="${_NVJOIN%:}"
  export LD_LIBRARY_PATH="$_NVJOIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "$DIR/.venv/bin/python" "$DIR/voice-daemon.py"
