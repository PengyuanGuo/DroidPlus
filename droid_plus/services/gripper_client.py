# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
from typing import Any

import requests

from droid_plus.constants import GRIPPER_SERVICE_URL


class GripperClient:
    """Thin HTTP client for gripper services.

    Works with either:
      - ``gripper_service.py`` (Robotiq 2F-85 on the workstation), or
      - ``franky_service.py`` Franka Hand endpoints on the NUC
        (same Robotiq-style bit API: 0=open … 255=closed).

    Point ``base_url`` / ``GRIPPER_SERVICE_URL`` at the chosen service.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_s: float = 5.0,
        async_timeout_s: float = 0.35,
        async_busy_backoff_s: float = 0.05,
    ):
        self.base_url = (base_url or GRIPPER_SERVICE_URL).rstrip("/")
        self.timeout_s = float(timeout_s)
        # Async (fire-and-forget) settings. These affect only *_async methods.
        self.async_timeout_s = float(async_timeout_s)
        self.async_busy_backoff_s = float(async_busy_backoff_s)

        # Async worker state (latest-wins).
        self._async_lock = threading.RLock()
        self._async_cv = threading.Condition(self._async_lock)
        self._async_thread: threading.Thread | None = None
        self._async_stop = False
        self._async_latest: tuple[str, dict[str, Any] | None, dict[str, Any] | None] | None = None
        self._async_last_error: str | None = None
        self._async_last_status_code: int | None = None

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _wait_param(self, wait: bool) -> dict[str, str]:
        # FastAPI parses bool query params like "true"/"false".
        return {"wait": "true" if bool(wait) else "false"}

    # ----------------------------
    # Async latest-wins worker
    # ----------------------------

    def _ensure_async_worker(self) -> None:
        with self._async_lock:
            if self._async_thread is not None and self._async_thread.is_alive():
                return
            self._async_stop = False
            t = threading.Thread(target=self._async_loop, name="GripperClientAsync", daemon=True)
            self._async_thread = t
            t.start()

    def _async_submit(self, method: str, path: str, *, params: dict[str, Any] | None, json: dict[str, Any] | None) -> None:
        """
        Submit an async request (latest-wins).

        - method: "GET" | "POST"
        - path: service path (e.g. "/open")
        """
        self._ensure_async_worker()
        with self._async_lock:
            self._async_latest = (method.upper(), {"path": str(path), "params": params}, {"json": json} if json is not None else None)
            self._async_cv.notify()

    def shutdown_async(self, *, join_timeout_s: float = 0.5) -> None:
        """Stop the async worker thread (best-effort)."""
        with self._async_lock:
            self._async_stop = True
            self._async_cv.notify_all()
            t = self._async_thread
        if t is not None:
            try:
                t.join(timeout=float(join_timeout_s))
            except Exception:
                pass

    def async_last_error(self) -> dict[str, Any]:
        """Return last async error info (for debugging)."""
        with self._async_lock:
            return {
                "last_error": self._async_last_error,
                "last_status_code": self._async_last_status_code,
                "worker_alive": bool(self._async_thread.is_alive()) if self._async_thread is not None else False,
            }

    def _async_loop(self) -> None:
        while True:
            with self._async_lock:
                while not self._async_stop and self._async_latest is None:
                    self._async_cv.wait(timeout=0.5)
                if self._async_stop:
                    return
                item = self._async_latest
                self._async_latest = None

            if item is None:
                continue

            method, meta, payload = item
            path = str((meta or {}).get("path", "/"))
            params = (meta or {}).get("params", None)
            json_payload = (payload or {}).get("json", None) if payload is not None else None

            try:
                if method == "POST":
                    r = requests.post(
                        self._url(path),
                        params=params,
                        json=json_payload,
                        timeout=self.async_timeout_s,
                    )
                else:
                    r = requests.get(
                        self._url(path),
                        params=params,
                        timeout=self.async_timeout_s,
                    )

                with self._async_lock:
                    self._async_last_status_code = int(getattr(r, "status_code", 0) or 0)

                # If gripper is busy, retry later, but only for the *latest* command.
                if int(r.status_code) == 409:
                    with self._async_lock:
                        # If a newer command arrived, drop this one.
                        has_newer = self._async_latest is not None
                    if not has_newer:
                        try:
                            threading.Event().wait(self.async_busy_backoff_s)
                        except Exception:
                            pass
                        # Requeue the same item (still latest because none newer arrived).
                        with self._async_lock:
                            if self._async_latest is None and not self._async_stop:
                                self._async_latest = item
                                self._async_cv.notify()
                    continue

                r.raise_for_status()
                with self._async_lock:
                    self._async_last_error = None
            except requests.HTTPError as e:
                st = None
                try:
                    st = int(e.response.status_code) if e.response is not None else None
                except Exception:
                    st = None
                with self._async_lock:
                    self._async_last_status_code = st
                    self._async_last_error = f"HTTPError: {repr(e)}"
                # On 409: handled above; any other HTTP error is dropped (latest-wins will send next command).
            except Exception as e:
                with self._async_lock:
                    self._async_last_error = f"{type(e).__name__}: {repr(e)}"

    def health(self) -> dict[str, Any]:
        r = requests.get(self._url("/health"), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def connect(self) -> dict[str, Any]:
        r = requests.post(self._url("/connect"), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def reset(self) -> dict[str, Any]:
        r = requests.post(self._url("/reset"), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def activate(self) -> dict[str, Any]:
        r = requests.post(self._url("/activate"), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def reset_activate(self) -> dict[str, Any]:
        r = requests.post(self._url("/reset_activate"), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def open(self, *, speed: int = 255, force: int = 255, wait: bool = True) -> dict[str, Any]:
        r = requests.post(
            self._url("/open"),
            params=self._wait_param(wait),
            json={"position": 0, "speed": int(speed), "force": int(force)},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def open_async(self, *, speed: int = 255, force: int = 255, wait: bool = False) -> None:
        """Fire-and-forget open command (latest-wins)."""
        self._async_submit(
            "POST",
            "/open",
            params=self._wait_param(wait),
            json={"position": 0, "speed": int(speed), "force": int(force)},
        )

    def close(self, *, speed: int = 255, force: int = 255, wait: bool = True) -> dict[str, Any]:
        r = requests.post(
            self._url("/close"),
            params=self._wait_param(wait),
            json={"position": 255, "speed": int(speed), "force": int(force)},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def close_async(self, *, speed: int = 255, force: int = 255, wait: bool = False) -> None:
        """Fire-and-forget close command (latest-wins)."""
        self._async_submit(
            "POST",
            "/close",
            params=self._wait_param(wait),
            json={"position": 255, "speed": int(speed), "force": int(force)},
        )

    def go_to(self, position: int, *, speed: int = 255, force: int = 255, wait: bool = True) -> dict[str, Any]:
        payload = {"position": int(position), "speed": int(speed), "force": int(force)}
        r = requests.post(self._url("/go_to"), params=self._wait_param(wait), json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def go_to_async(self, position: int, *, speed: int = 255, force: int = 255, wait: bool = False) -> None:
        """Fire-and-forget go_to command (latest-wins)."""
        payload = {"position": int(position), "speed": int(speed), "force": int(force)}
        self._async_submit("POST", "/go_to", params=self._wait_param(wait), json=payload)

    def go_to_mm(self, position_mm: float, *, speed: int = 255, force: int = 255, wait: bool = True) -> dict[str, Any]:
        payload = {"position_mm": float(position_mm), "speed": int(speed), "force": int(force)}
        r = requests.post(self._url("/go_to_mm"), params=self._wait_param(wait), json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def go_to_mm_async(self, position_mm: float, *, speed: int = 255, force: int = 255, wait: bool = False) -> None:
        """Fire-and-forget go_to_mm command (latest-wins)."""
        payload = {"position_mm": float(position_mm), "speed": int(speed), "force": int(force)}
        self._async_submit("POST", "/go_to_mm", params=self._wait_param(wait), json=payload)

    def calibrate(self, *, close_mm: float, open_mm: float) -> dict[str, Any]:
        r = requests.post(
            self._url("/calibrate"),
            json={"close_mm": float(close_mm), "open_mm": float(open_mm)},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def is_activated(self) -> dict[str, Any]:
        r = requests.get(self._url("/is_activated"), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def is_calibrated(self) -> dict[str, Any]:
        r = requests.get(self._url("/is_calibrated"), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def position(self) -> dict[str, Any]:
        r = requests.get(self._url("/position"), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def position_mm(self) -> dict[str, Any]:
        r = requests.get(self._url("/position_mm"), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def status(self) -> dict[str, Any]:
        r = requests.get(self._url("/status"), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def gripper_state(self, *, closed_threshold: int = 128) -> dict[str, Any]:
        r = requests.get(
            self._url("/gripper_state"),
            params={"closed_threshold": int(closed_threshold)},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def gripper_position_frac(self) -> float:
        """
        Return current gripper position normalized to [0,1] (position_bits / 255).
        """
        st = self.gripper_state()
        if "position_frac" in st:
            return float(st["position_frac"])
        pos = float(st.get("position_bits", 0.0))
        return max(0.0, min(1.0, pos / 255.0))


if __name__ == "__main__":
    c = GripperClient()
    print("connect:", c.connect())
    print("activate:", c.activate())
    print("open:", c.open())
    print("close:", c.close())
    print("open:", c.open())
    print("status:", c.status())
