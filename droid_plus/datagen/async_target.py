"""Post the latest joint target from a worker so control never waits on HTTP."""
from __future__ import annotations

import threading
import time

import numpy as np


class AsyncTargetSender:
    """Latest-value mailbox to ``droid.set_target_joint_state``.

    Targets submitted while a post is in flight replace each other; only the
    newest is sent. ``check`` raises if a post failed or has been in flight for
    longer than ``stall_s`` (``first_stall_s`` for the first post, which starts
    the NUC's motion and is slower).
    """
    def __init__(self, droid, *, stall_s=0.5, first_stall_s=2.0):
        self.droid = droid
        self.stall_s, self.first_stall_s = stall_s, first_stall_s
        self.condition = threading.Condition()
        self.latest = None
        self.stopping = False
        self.error = None
        self.in_flight_since = None
        self.posts = 0
        self.slowest_ms = 0.0
        self.thread = threading.Thread(target=self._run, name='teleop-target-sender', daemon=True)
        self.thread.start()

    def submit(self, q, seq):
        self.check()
        with self.condition:
            self.latest = (np.asarray(q, dtype=float).copy(), seq)
            self.condition.notify()

    def check(self):
        with self.condition:
            if self.error is not None:
                raise RuntimeError('Robot target post failed') from self.error
            started = self.in_flight_since
            limit = self.stall_s if self.posts else self.first_stall_s
        if started is not None and time.monotonic() - started > limit:
            raise RuntimeError(f'Robot target post in flight for over {limit * 1000:.0f} ms; episode stopped')

    def _run(self):
        try:
            while True:
                with self.condition:
                    while self.latest is None and not self.stopping:
                        self.condition.wait()
                    if self.stopping:
                        return
                    (q, seq), self.latest = self.latest, None
                    self.in_flight_since = started = time.monotonic()
                self.droid.set_target_joint_state(q, velocities=[0.0] * 7, seq=seq)
                elapsed_ms = (time.monotonic() - started) * 1000
                with self.condition:
                    self.in_flight_since = None
                    self.posts += 1
                    self.slowest_ms = max(self.slowest_ms, elapsed_ms)
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.in_flight_since = None

    def take_stats(self):
        """Return ``(posts, slowest_ms)`` since the previous call."""
        with self.condition:
            stats = (self.posts, self.slowest_ms)
            self.slowest_ms = 0.0
            return stats

    def close(self, timeout=6.0):
        """Drop any unsent target and wait for an in-flight post to finish."""
        with self.condition:
            self.stopping = True
            self.latest = None
            self.condition.notify()
        self.thread.join(timeout=timeout)
