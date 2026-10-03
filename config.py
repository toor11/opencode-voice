#!/usr/bin/env python3
"""Configuration for opencode-voice: TOML file + environment overrides.

Lookup order (first hit wins per key): environment ``OC_VOICE_*`` >
``~/.config/opencode-voice/config.toml`` (or ``$XDG_CONFIG_HOME``) > built-in
defaults. There is no required setup step: with no config file the daemon,
client and benchmark all run on defaults. See config.example.toml for every
key (server, audio, transcription incl. confidence, vad, streaming,
normalization vocabulary, cancellation).

Socket location: ``[server] socket = "auto"`` (default) resolves to
``$XDG_RUNTIME_DIR/opencode-voice.sock`` when that directory exists, else
``/tmp/oc-voice/voice.sock`` (the pre-existing spool dir). Set an explicit
path in config or ``OC_VOICE_SOCKET`` to override. Never TCP.
"""

from __future__ import annotations

import os
from copy import deepcopy

DEFAULTS = {
    "server": {
        "socket": "auto",
        "legacy_file_mode": False,
        "spool_dir": "/tmp/oc-voice",
    },
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "chunk_ms": 100,          # mic -> socket frame size
    },
    "transcription": {
        "model": "small.en",
        "device": "auto",         # auto | cuda | cpu
        "compute_type": "int8",
        "language": "en",
        "beam_size": 1,
        "confidence_enabled": True,
        # Below this the result is still returned, but flagged low-confidence
        # (never silently discarded, never auto "corrected" by an LLM).
        "low_confidence_threshold": 0.55,
        "initial_prompt": (
            "OpenCode on Garuda Linux with Hyprland. "
            "Dictation in the terminal, browser and documents."
        ),
    },
    "vad": {
        "enabled": True,
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 200,
        "min_speech_s": 0.25,    # below this the clip counts as silence
        "trim_savings_s": 0.5,   # rewrite wav/buffer only if it saves >= this
    },
    "streaming": {
        "enabled": True,       # False = buffer everything, single final only
        "interval_ms": 2000,     # how often to attempt a partial transcript
        "window_s": 8.0,         # partials transcribe at most this much audio
        "min_new_speech_s": 1.0, # ... and only if this much new speech arrived
    },
    "normalization": {
        "enabled": True,
        # Conservative misrecognition pairs only (multi-word). A user list
        # replaces these defaults wholesale. See normalizer.py.
        "replaces": [
            ["open code", "OpenCode"],
            ["get hub", "GitHub"],
            ["get lab", "GitLab"],
            ["hip land", "Hyprland"],
            ["hi pr land", "Hyprland"],
            ["dock er", "Docker"],
            ["cue wen", "Qwen"],
            ["q wen", "Qwen"],
            ["key me", "Kimi"],
            ["post gress", "PostgreSQL"],
            ["postgress", "PostgreSQL"],
            ["my sequel", "MySQL"],
            ["node j s", "Node.js"],
        ],
    },
    "cancellation": {
        "enabled": True,  # False = CANCEL behaves like STOP (still finalizes)
    },
}

# env var -> (section, key, type)
ENV_OVERRIDES = {
    "OC_VOICE_SOCKET": ("server", "socket", str),
    "OC_VOICE_SPOOL": ("server", "spool_dir", str),
    "OC_VOICE_LEGACY": ("server", "legacy_file_mode", lambda v: v == "1"),
    "OC_VOICE_MODEL": ("transcription", "model", str),
    "OC_VOICE_DEVICE": ("transcription", "device", str),
    "OC_VOICE_LANGUAGE": ("transcription", "language", str),
    "OC_VOICE_CHUNK_MS": ("audio", "chunk_ms", int),
    "OC_VOICE_STREAM_MS": ("streaming", "interval_ms", int),
    "OC_VOICE_VAD_THRESHOLD": ("vad", "threshold", float),
    "OC_VOICE_VAD_MIN_SPEECH_MS": ("vad", "min_speech_duration_ms", int),
    "OC_VOICE_VAD_MIN_SILENCE_MS": ("vad", "min_silence_duration_ms", int),
    "OC_VOICE_VAD_PAD_MS": ("vad", "speech_pad_ms", int),
    "OC_VOICE_CONFIDENCE": ("transcription", "confidence_enabled",
                             lambda v: v == "1"),
    "OC_VOICE_LOW_CONF": ("transcription", "low_confidence_threshold", float),
    "OC_VOICE_NORMALIZE": ("normalization", "enabled", lambda v: v == "1"),
}


def config_path() -> str:
    base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return os.path.join(base, "opencode-voice", "config.toml")


def load(path: str | None = None) -> dict:
    cfg = deepcopy(DEFAULTS)
    path = path or config_path()
    try:
        import tomllib

        with open(path, "rb") as f:
            user = tomllib.load(f)
        for section, values in user.items():
            if section in cfg and isinstance(values, dict):
                cfg[section].update(values)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"config: ignoring {path} ({e}), using defaults")
    for env, (section, key, conv) in ENV_OVERRIDES.items():
        if env in os.environ:
            try:
                cfg[section][key] = conv(os.environ[env])
            except ValueError:
                print(f"config: ignoring invalid {env}={os.environ[env]!r}")
    return cfg


def resolve_socket(cfg: dict) -> str:
    """Socket path: explicit config wins, else XDG runtime dir, else spool dir."""
    sock = cfg["server"]["socket"]
    if sock and sock != "auto":
        return sock
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime and os.path.isdir(runtime):
        return os.path.join(runtime, "opencode-voice.sock")
    spool = cfg["server"]["spool_dir"]
    os.makedirs(spool, exist_ok=True)
    return os.path.join(spool, "voice.sock")
