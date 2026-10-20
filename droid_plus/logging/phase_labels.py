"""Operator phase selection and review of per-frame labels."""
from __future__ import annotations

import threading


class PhaseSelector:
    """Thread-safe phase state; create a fresh selector for every episode."""

    def __init__(self) -> None:
        self._phase = 0
        self._lock = threading.Lock()

    def get(self) -> int:
        with self._lock:
            return self._phase

    def set(self, phase: int) -> None:
        if type(phase) is not int or not 0 <= phase <= 127:
            raise ValueError("phase_index must be an integer in [0, 127]")
        with self._lock:
            self._phase = phase


def review_phase_labels(labels: list[int], *, min_frames: int = 1) -> dict:
    """Flag sequence errors without changing the operator's recorded labels."""
    issues = []
    segments = []
    for i, phase in enumerate(labels):
        if not segments or segments[-1]["phase_index"] != phase:
            if not segments and phase != 0:
                issues.append("First recorded phase is not 0")
            if segments and phase != segments[-1]["phase_index"] + 1:
                issues.append(f"Frame {i}: phase {segments[-1]['phase_index']} -> {phase}; expected +1")
            segments.append({"phase_index": phase, "start_frame": i, "frames": 0})
        segments[-1]["frames"] += 1
    if not labels:
        issues.append("No phase-labelled frames recorded")
    for segment in segments:
        if segment["frames"] < min_frames:
            issues.append(f"Phase {segment['phase_index']} at frame {segment['start_frame']} is shorter than {min_frames} frames")
    return {"needs_review": bool(issues), "issues": issues, "segments": segments, "min_frames": min_frames}
