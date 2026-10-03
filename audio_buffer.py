#!/usr/bin/env python3
"""Bounded thread-safe audio buffer: O(1) append, cheap tail, one full copy.

Replaces the old pattern of ``np.concatenate(all_chunks)`` on every partial
(which recopied the whole recording each time). Layout: a deque of float32
mono 16kHz NumPy chunks plus a running sample count.

* ``append``: O(1), never blocks on ASR.
* ``tail(seconds)``: walks from the newest chunk, concatenates only what is
  needed for the partial window.
* ``full()``: single concatenation, called once at STOP for the final.
* bounded: chunks older than ``max_seconds`` are dropped (default 120s --
  far beyond any utterance; prevents unbounded growth on a stuck session).
* locks: short critical sections only; callers MUST copy out and release
  before running Whisper (never hold the buffer lock during GPU inference).
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np


class AudioBuffer:
    def __init__(self, sample_rate: int = 16000, max_seconds: float = 120.0):
        self.sr = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self._chunks: deque = deque()
        self._total = 0
        self._lock = threading.Lock()

    def append(self, audio: np.ndarray) -> None:
        arr = np.asarray(audio, dtype=np.float32).ravel()
        if not len(arr):
            return
        with self._lock:
            self._chunks.append(arr)
            self._total += len(arr)
            # Bound memory: drop oldest while over budget.
            while self._total > self.max_samples and self._chunks:
                old = self._chunks.popleft()
                self._total -= len(old)

    def __len__(self) -> int:
        with self._lock:
            return self._total

    @property
    def seconds(self) -> float:
        return len(self) / self.sr

    def tail(self, seconds: float) -> np.ndarray:
        """Newest ``seconds`` of audio. Copies only the chunks it needs."""
        need = int(seconds * self.sr)
        with self._lock:
            if not self._chunks or need <= 0:
                return np.zeros(0, np.float32)
            parts = []
            remaining = min(need, self._total)
            for chunk in reversed(self._chunks):
                take = min(len(chunk), remaining)
                parts.append(chunk[len(chunk) - take:])
                remaining -= take
                if remaining <= 0:
                    break
            parts.reverse()
            return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def full(self) -> np.ndarray:
        """Entire recording. Single copy; call once at STOP."""
        with self._lock:
            if not self._chunks:
                return np.zeros(0, np.float32)
            return np.concatenate(self._chunks)
