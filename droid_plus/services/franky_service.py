# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
FR3 Franky joint-state service (streaming joint position+velocity targets via JointMotion).

Run (single worker):
  uvicorn service:app --host 0.0.0.0 --port 54321 --workers 1

Example client:
  import requests
  nuc = "http://NUC_IP:54321"
  requests.post(f"{nuc}/target_joint_state", json={
      "positions": [0.0]*7,
      "velocities": [0.0]*7,
      "seq": 1,
  })
  requests.post(f"{nuc}/stop")

Env:
  FRANKY_ROBOT_IP=<robot-ip-or-hostname>
  CONTROL_HZ=50
  COMMAND_TIMEOUT_S=0.5
  FRANKA_GRIPPER=1                 # set 0 to disable Franka Hand init
  GRIPPER_FOLLOW_MODE=intent       # intent | binary | waypoint (see _gripper_command_loop)
  GRIPPER_SETTLE_S=0.15            # leader considered still after this long without change
  GRIPPER_DEADBAND_M=0.004         # settled width error tolerated (m)
  GRIPPER_STOP_LEAD_M=0.004        # stop() this far before target to absorb latency (m)
  ALLOWED_CLIENT_IP=<client-ip>      # optional allowlist (direct-connect only)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

try:
    import franky
except ImportError as exc:
    raise ImportError(
        "franky-service requires a separately installed copy of franky-control. "
        "DROID+ does not distribute or automatically install it; see README.md "
        "for the upstream license notice and installation instructions."
    ) from exc

from droid_plus.services.franka_gripper import (
    DEFAULT_FRANKA_HAND_MAX_WIDTH_M,
    bits_to_speed_m_s,
    bits_to_width_m,
    width_m_to_bits,
)

LOG = logging.getLogger("franky_service")

# CONSTANTS
N_JOINTS = 7
DEFAULT_CONTROL_HZ = 50
DEFAULT_COMMAND_TIMEOUT_S = 10.0
DEFAULT_ALLOWED_CLIENT_IP = ""
DEFAULT_FRANKY_ROBOT_IP = "localhost"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SERVICE_PORT = 54321

EFFECTIVE_FLOAT_INF_SECONDS = 1_000_000 # This is about 2 weeks

# Joint-space \"home\" pose (matches `franky_client.HOME_POSITION`).
HOME_POSITION: list[float] = [0.0, -0.40, 0.0, -1.9, 0.0, 1.5, 0.0]


LANDING_PAGE_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>franky_service</title>
    <style>
      :root { color-scheme: dark; }
      body { margin: 24px; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: #0b0f14; color: #e6edf3; }
      .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
      .card { background: #0f1620; border: 1px solid #223042; border-radius: 10px; padding: 14px 14px; min-width: 320px; flex: 1; }
      .title { font-size: 16px; font-weight: 700; margin: 0 0 8px; }
      .muted { color: #9fb2c8; }
      pre { margin: 8px 0 0; padding: 10px; background: #0b111a; border: 1px solid #223042; border-radius: 8px; overflow: auto; max-height: 420px; }
      code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; }
      button { background: #1f6feb; color: white; border: none; padding: 8px 12px; border-radius: 8px; cursor: pointer; }
      button.secondary { background: #223042; }
      button:disabled { opacity: 0.6; cursor: not-allowed; }
      input { background: #0b111a; color: #e6edf3; border: 1px solid #223042; border-radius: 8px; padding: 8px 10px; width: 140px; }
      .kv { display: grid; grid-template-columns: 190px 1fr; gap: 6px 10px; font-size: 13px; }
      .ok { color: #3fb950; }
      .bad { color: #f85149; }
      .plot-wrap { width: 100%%; }
      canvas { width: 100%%; height: 320px; background: #0b111a; border: 1px solid #223042; border-radius: 8px; }
      select { background: #0b111a; color: #e6edf3; border: 1px solid #223042; border-radius: 8px; padding: 8px 10px; }
      .legend { display: flex; gap: 14px; flex-wrap: wrap; align-items: center; font-size: 12px; color: #9fb2c8; margin-top: 10px; }
      .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }
    </style>
  </head>
  <body>
    <div class="row">
      <div class="card">
        <div class="title">franky_service</div>
        <div class="muted">Simple status dashboard (auto-refresh).</div>
        <div style="height: 10px"></div>
        <div class="kv">
          <div class="muted">robot_ip</div><div><code id="robot_ip">...</code></div>
          <div class="muted">robot_connected</div><div><code id="robot_connected">...</code></div>
          <div class="muted">stop_latched</div><div><code id="stop_latched">...</code></div>
          <div class="muted">has_target</div><div><code id="has_target">...</code></div>
          <div class="muted">command_timeout_s</div><div><code id="command_timeout_s">...</code></div>
          <div class="muted">last_control_error</div><div><code id="last_control_error">...</code></div>
        </div>
        <div style="height: 12px"></div>
        <div class="row" style="gap: 10px; align-items: center;">
          <button id="stop_btn" onclick="doStop()">Stop</button>
          <button class="secondary" id="go_home_btn" onclick="goHome()">Go home</button>
          <input id="timeout_input" type="number" step="0.1" min="0.001" placeholder="timeout_s" />
          <button class="secondary" onclick="setTimeoutS()">Set timeout</button>
          <button class="secondary" onclick="setTimeoutInfinity()">Set ∞</button>
        </div>
      </div>
      <div class="card">
        <div class="title">current_joint_state</div>
        <div class="muted">From <code>/joint_state</code></div>
        <pre><code id="joint_state">Loading...</code></pre>
      </div>
      <div class="card">
        <div class="title">latest_target_joint_state</div>
        <div class="muted">From <code>/target_joint_state</code></div>
        <pre><code id="target_state">Loading...</code></pre>
      </div>
      <div class="card plot-wrap">
        <div class="row" style="justify-content: space-between; align-items: center;">
          <div>
            <div class="title">joint positions (actual vs target)</div>
            <div class="muted">Live plot from <code>/joint_state</code> and <code>/target_joint_state</code></div>
          </div>
          <div class="row" style="gap: 10px; align-items: center;">
            <div class="muted">joint</div>
            <select id="joint_select"></select>
            <div class="muted">horizon</div>
            <input id="horizon_input" type="number" step="1" min="1" placeholder="seconds" />
            <button class="secondary" onclick="setHorizon()">Set</button>
          </div>
        </div>
        <div style="height: 10px"></div>
        <canvas id="joint_plot" width="900" height="320"></canvas>
        <div class="legend" id="plot_legend"></div>
      </div>
    </div>
    <script>
      const robotIp = %(robot_ip_json)s;
      const N_JOINTS = 7;

      function setText(id, val, cls=null) {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = (val === undefined || val === null) ? "" : String(val);
        el.className = cls ? cls : "";
      }

      function setJson(id, obj) {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = JSON.stringify(obj, null, 2);
      }

      async function fetchJson(path) {
        const r = await fetch(path, { cache: "no-store" });
        const txt = await r.text();
        let data;
        try { data = JSON.parse(txt); } catch { data = { raw: txt }; }
        if (!r.ok) throw { status: r.status, data };
        return data;
      }

      // ----------------------------
      // Status refresh (1 Hz)
      // ----------------------------

      async function refreshStatusOnce() {
        setText("robot_ip", robotIp);
        try {
          const [health, timeout, joint, target] = await Promise.all([
            fetchJson("/health"),
            fetchJson("/command_timeout"),
            fetchJson("/joint_state").catch(e => ({ error: e })),
            fetchJson("/target_joint_state").catch(e => ({ error: e })),
          ]);

          setText("robot_connected", health.robot_connected, health.robot_connected ? "ok" : "bad");
          setText("stop_latched", health.stop_latched);
          setText("has_target", health.has_target);
          setText("last_control_error", health.last_control_error || "");
          setText("command_timeout_s", timeout.command_timeout_s);

          setJson("joint_state", joint);
          setJson("target_state", target);
        } catch (e) {
          setText("last_control_error", (e && e.data) ? JSON.stringify(e.data) : String(e));
        }
      }

      // ----------------------------
      // Plot (data at 10 Hz; draw on updates)
      // ----------------------------

      const plot = {
        canvas: null,
        ctx: null,
        // history: arrays of {t, v} values per joint
        t0: performance.now(),
        sampleHz: 10,
        horizonS: 30,          // displayed time window
        bufferS: 600,          // keep up to 10 minutes of data for instant horizon changes
        maxPoints: 6000,       // derived from bufferS * sampleHz
        selectedJoint: 0,
        actual: Array.from({ length: N_JOINTS }, () => []),
        target: Array.from({ length: N_JOINTS }, () => []),
        colors: ["#58a6ff", "#3fb950", "#f778ba", "#ffa657", "#a371f7", "#ff7b72", "#9fb2c8"],
        lastActual: Array.from({ length: N_JOINTS }, () => null),
        lastTarget: Array.from({ length: N_JOINTS }, () => null),
      };

      function initJointSelect() {
        const sel = document.getElementById("joint_select");
        if (!sel) return;
        sel.innerHTML = "";
        for (let j = 0; j < N_JOINTS; j++) {
          const opt = document.createElement("option");
          opt.value = String(j);
          opt.textContent = `q${j}`;
          sel.appendChild(opt);
        }
        sel.value = String(plot.selectedJoint);
        sel.addEventListener("change", () => {
          plot.selectedJoint = Number(sel.value);
          drawPlot();
        });
      }

      function initHorizonInput() {
        const inp = document.getElementById("horizon_input");
        if (!inp) return;
        inp.value = String(plot.horizonS);
      }

      function setHorizon() {
        const inp = document.getElementById("horizon_input");
        if (!inp) return;
        const v = Number(inp.value);
        if (!Number.isFinite(v) || v <= 0) return;
        plot.horizonS = Math.max(1, Math.floor(v));
        drawPlot();
        updateLegend();
      }

      function pushPoint(series, t, v) {
        series.push({ t, v });
        if (series.length > plot.maxPoints) series.splice(0, series.length - plot.maxPoints);
      }

      function getSeriesRange(series) {
        let minV = Infinity, maxV = -Infinity;
        for (const p of series) {
          if (!Number.isFinite(p.v)) continue;
          if (p.v < minV) minV = p.v;
          if (p.v > maxV) maxV = p.v;
        }
        if (!Number.isFinite(minV) || !Number.isFinite(maxV)) return null;
        if (minV === maxV) {
          const pad = Math.max(0.1, Math.abs(minV) * 0.05);
          return { minV: minV - pad, maxV: maxV + pad };
        }
        const pad = (maxV - minV) * 0.08;
        return { minV: minV - pad, maxV: maxV + pad };
      }

      function fmt(v) {
        if (v === null || v === undefined) return "";
        if (!Number.isFinite(v)) return "";
        return v.toFixed(3);
      }

      function updateLegend() {
        const el = document.getElementById("plot_legend");
        if (!el) return;
        const j = plot.selectedJoint;
        const c = plot.colors[j %% plot.colors.length];
        const a = plot.lastActual[j];
        const t = plot.lastTarget[j];
        el.innerHTML = `
          <span><span class="swatch" style="background:${c}"></span>joint q${j}</span>
          <span><b>actual</b>: ${fmt(a)} rad</span>
          <span><b>target</b>: ${fmt(t)} rad</span>
          <span class="muted">(solid=actual, dashed=target, horizon=${plot.horizonS}s)</span>
        `;
      }

      function drawAxes(ctx, w, h, minV, maxV, tMin, tMax) {
        const padL = 52, padR = 14, padT = 10, padB = 26;
        const x0 = padL, y0 = padT, x1 = w - padR, y1 = h - padB;

        // background
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#0b111a";
        ctx.fillRect(0, 0, w, h);

        // grid + border
        ctx.strokeStyle = "#223042";
        ctx.lineWidth = 1;
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);

        ctx.fillStyle = "#9fb2c8";
        ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";

        // y ticks
        const yTicks = 5;
        for (let i = 0; i <= yTicks; i++) {
          const a = i / yTicks;
          const y = y1 - a * (y1 - y0);
          const v = minV + a * (maxV - minV);
          ctx.strokeStyle = "#192434";
          ctx.beginPath();
          ctx.moveTo(x0, y);
          ctx.lineTo(x1, y);
          ctx.stroke();
          ctx.fillStyle = "#9fb2c8";
          ctx.fillText(v.toFixed(2), 6, y + 4);
        }

        // x ticks (seconds)
        const xTicks = 5;
        for (let i = 0; i <= xTicks; i++) {
          const a = i / xTicks;
          const x = x0 + a * (x1 - x0);
          const t = tMin + a * (tMax - tMin);
          ctx.strokeStyle = "#192434";
          ctx.beginPath();
          ctx.moveTo(x, y0);
          ctx.lineTo(x, y1);
          ctx.stroke();
          ctx.fillStyle = "#9fb2c8";
          ctx.fillText(`${t.toFixed(1)}s`, x - 14, h - 8);
        }

        return { x0, y0, x1, y1 };
      }

      function drawSeries(ctx, box, series, tMin, tMax, vMin, vMax, color, dashed=false) {
        if (!series || series.length < 2) return;
        const { x0, y0, x1, y1 } = box;
        const xScale = (x1 - x0) / Math.max(1e-9, (tMax - tMin));
        const yScale = (y1 - y0) / Math.max(1e-9, (vMax - vMin));
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash(dashed ? [7, 6] : []);
        ctx.beginPath();
        let started = false;
        for (const p of series) {
          const t = p.t;
          const v = p.v;
          if (!Number.isFinite(t) || !Number.isFinite(v)) continue;
          if (t < tMin || t > tMax) continue;
          const x = x0 + (t - tMin) * xScale;
          const y = y1 - (v - vMin) * yScale;
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      function drawPlot() {
        if (!plot.ctx || !plot.canvas) return;
        const j = plot.selectedJoint;
        const aSeries = plot.actual[j];
        const tSeries = plot.target[j];
        const combined = aSeries.concat(tSeries);

        // Use a sliding time window based on oldest/newest in combined series
        if (combined.length < 2) return;
        const tMax = combined[combined.length - 1].t;
        const tMin = Math.max(combined[0].t, tMax - plot.horizonS);
        const rA = getSeriesRange(aSeries);
        const rT = getSeriesRange(tSeries);
        const r = (rA && rT)
          ? { minV: Math.min(rA.minV, rT.minV), maxV: Math.max(rA.maxV, rT.maxV) }
          : (rA || rT);
        if (!r) return;

        const w = plot.canvas.width;
        const h = plot.canvas.height;
        const box = drawAxes(plot.ctx, w, h, r.minV, r.maxV, tMin, tMax);
        const color = plot.colors[j %% plot.colors.length];
        drawSeries(plot.ctx, box, aSeries, tMin, tMax, r.minV, r.maxV, color, false);
        drawSeries(plot.ctx, box, tSeries, tMin, tMax, r.minV, r.maxV, color, true);
        updateLegend();
      }

      async function refreshPlotOnce() {
        try {
          const [joint, target] = await Promise.all([
            fetchJson("/joint_state"),
            fetchJson("/target_joint_state"),
          ]);

          const now = performance.now();
          const t = (now - plot.t0) / 1000.0;
          const posA = (joint && Array.isArray(joint.positions)) ? joint.positions : null;
          const posT = (target && Array.isArray(target.positions)) ? target.positions : null;

          if (posA && posA.length >= N_JOINTS) {
            for (let j = 0; j < N_JOINTS; j++) {
              const v = Number(posA[j]);
              if (Number.isFinite(v)) {
                pushPoint(plot.actual[j], t, v);
                plot.lastActual[j] = v;
              }
            }
          }

          if (posT && posT.length >= N_JOINTS) {
            for (let j = 0; j < N_JOINTS; j++) {
              const v = Number(posT[j]);
              if (Number.isFinite(v)) {
                pushPoint(plot.target[j], t, v);
                plot.lastTarget[j] = v;
              }
            }
          }

          drawPlot();
        } catch (e) {
          // ignore transient errors (e.g., robot disconnected)
        }
      }

      async function doStop() {
        const btn = document.getElementById("stop_btn");
        if (btn) btn.disabled = true;
        try {
          await fetch("/stop", { method: "POST" });
        } finally {
          if (btn) btn.disabled = false;
          await refreshOnce();
        }
      }

      async function goHome() {
        const btn = document.getElementById("go_home_btn");
        if (btn) btn.disabled = true;
        try {
          await fetch("/go_home", { method: "POST" });
        } finally {
          if (btn) btn.disabled = false;
          await refreshOnce();
        }
      }

      async function setTimeoutS() {
        const raw = document.getElementById("timeout_input").value;
        if (!raw) return;
        const v = Number(raw);
        if (!Number.isFinite(v) || v <= 0) return;
        await fetch("/command_timeout", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ command_timeout_s: v }),
        });
        await refreshOnce();
      }

      async function setTimeoutInfinity() {
        // JSON has no Infinity. Use a very large timeout.
        await fetch("/command_timeout", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ command_timeout_s: 1e12 }),
        });
        await refreshOnce();
      }

      function refreshOnce() {
        // Backwards compatibility for button handlers (stop / set timeout),
        // but keep status refresh separate from plot refresh loops.
        return refreshStatusOnce();
      }

      // init
      plot.canvas = document.getElementById("joint_plot");
      plot.ctx = plot.canvas ? plot.canvas.getContext("2d") : null;
      plot.maxPoints = Math.max(10, Math.floor(plot.bufferS * plot.sampleHz));
      initJointSelect();
      initHorizonInput();
      updateLegend();
      refreshStatusOnce();
      refreshPlotOnce();

      // loops
      setInterval(refreshStatusOnce, 1000);
      setInterval(refreshPlotOnce, Math.floor(1000 / plot.sampleHz));
    </script>
  </body>
</html>
"""


def _get_env_float(name: str, default: float) -> float:
    """Read float env var with a default."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _get_env_int(name: str, default: int) -> int:
    """Read int env var with a default."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _validate_vec(name: str, values: list[float], n: int = N_JOINTS) -> None:
    """Validate a numeric vector payload."""
    if len(values) != n:
        raise HTTPException(status_code=400, detail=f"{name} must have length {n}")
    for v in values:
        if not isinstance(v, (float, int)):
            raise HTTPException(status_code=400, detail=f"{name} must contain numbers")
        fv = float(v)
        if not (fv == fv) or fv in (float("inf"), float("-inf")):
            raise HTTPException(status_code=400, detail=f"{name} must contain finite values (no NaN/Inf)")


# DATA MODELS

class JointTargetIn(BaseModel):
    positions: list[float] = Field(..., description="Joint positions (rad), length 7")
    velocities: list[float] = Field(..., description="Joint velocities (rad/s), length 7")
    seq: Optional[int] = Field(default=None, description="Optional client sequence number")


class JointTargetOut(BaseModel):
    positions: list[float]
    velocities: list[float]
    seq: Optional[int]
    accepted_timestamp_s: float
    age_s: float
    stop_latched: bool


class StopOut(BaseModel):
    stopped: bool
    timestamp_s: float


class CommandTimeoutIn(BaseModel):
    command_timeout_s: float = Field(..., gt=0.0, description="Stop if target is older than this (seconds).")


class CommandTimeoutOut(BaseModel):
    command_timeout_s: float



# INTERNAL STATE

@dataclass
class LatestTarget:
    positions: list[float]
    velocities: list[float]
    seq: Optional[int]
    accepted_timestamp_s: float


@dataclass
class AppState:
    robot: object | None
    lock: threading.Lock
    latest_target: LatestTarget | None
    stop_latched: bool
    shutdown: threading.Event
    control_thread: threading.Thread | None
    last_control_error: str | None
    command_timeout_s: float
    # Franka Hand (optional; shares robot IP, separate libfranka connection).
    # Homing follows GELLO (home once, Move-only, no Grasp). Streaming does NOT:
    # see _gripper_command_loop for why (Move is point-to-point; every retarget
    # costs a stop()).
    gripper: object | None = None
    gripper_lock: threading.Lock = field(default_factory=threading.Lock)
    gripper_busy: bool = False
    gripper_homed: bool = False
    gripper_last_error: str | None = None
    gripper_max_width_m: float = DEFAULT_FRANKA_HAND_MAX_WIDTH_M
    gripper_motion_thread: threading.Thread | None = None
    gripper_motion_gen: int = 0  # generation counter: stale move threads must not clear busy
    gripper_cmd_thread: threading.Thread | None = None
    # Latest leader target + intent.
    gripper_target_width_m: float | None = None
    gripper_target_speed_m_s: float = 0.1
    gripper_target_t_s: float = 0.0  # wall time of last target change
    gripper_target_dir: int = 0  # +1 opening, -1 closing (leader direction)
    # What the Hand is currently heading to (None = idle / unknown).
    gripper_goal_width_m: float | None = None
    gripper_last_cmd_width_m: float | None = None
    gripper_width_eps_m: float = 0.002  # ~6 bits at 80 mm
    # Follow tuning — env-overridable (GRIPPER_FOLLOW_MODE, GRIPPER_SETTLE_S,
    # GRIPPER_DEADBAND_M, GRIPPER_STOP_LEAD_M, GRIPPER_BINARY_CLOSE_FRAC/OPEN_FRAC).
    gripper_follow_mode: str = "intent"  # intent | binary | waypoint
    gripper_settle_s: float = 0.15  # leader "still" if target unchanged this long
    gripper_deadband_m: float = 0.004  # ignore settled errors below this
    gripper_stop_lead_m: float = 0.004  # stop() this early to absorb command latency
    gripper_binary_close_frac: float = 0.6  # binary mode: close when leader ≥ 60% closed
    gripper_binary_open_frac: float = 0.4  # binary mode: open when leader ≤ 40% closed



class GripperGoToBitsIn(BaseModel):
    """Robotiq-compatible gripper command (used by GripperClient / teleop)."""
    position: int = Field(0, ge=0, le=255, description="0=open, 255=closed")
    speed: int = Field(255, ge=0, le=255)
    force: int = Field(255, ge=0, le=255)


# INTERNAL FUNCTIONS

def _require_robot(st: AppState):
    """Return robot handle or raise 503."""
    if st.robot is None:
        raise HTTPException(status_code=503, detail="robot not available (is franky-control installed and robot reachable?)")
    return st.robot


def _require_gripper(st: AppState):
    """Return Franka Hand handle or raise 503."""
    if st.gripper is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Franka Hand not available",
                "last_error": st.gripper_last_error,
                "hint": "Ensure FRANKA_GRIPPER=1 and franky.Gripper(robot_ip) can connect",
            },
        )
    return st.gripper


def _gripper_enabled() -> bool:
    raw = os.getenv("FRANKA_GRIPPER", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _read_gripper_max_width_m(gripper: object) -> float:
    for attr in ("max_width", "maxWidth"):
        try:
            val = getattr(gripper, attr)
            if callable(val):
                val = val()
            w = float(val)
            if w > 0.0:
                return w
        except Exception:
            continue
    return DEFAULT_FRANKA_HAND_MAX_WIDTH_M


def _read_gripper_width_m(gripper: object) -> float:
    for attr in ("width",):
        try:
            val = getattr(gripper, attr)
            if callable(val):
                val = val()
            return float(val)
        except Exception:
            continue
    raise RuntimeError("unable to read Franka Hand width")


def _gripper_stop_best_effort(gripper: object) -> None:
    try:
        gripper.stop()
    except Exception:
        pass


def _release_gripper(st: AppState) -> bool:
    """Drop the libfranka Gripper connection so Desk regains the Hand card.

    Returns True if a connection was released.
    """
    with st.gripper_lock:
        g = st.gripper
        st.gripper = None
        st.gripper_busy = False
        st.gripper_homed = False
        st.gripper_target_width_m = None
        st.gripper_target_dir = 0
        st.gripper_goal_width_m = None
        st.gripper_last_cmd_width_m = None
        st.gripper_motion_gen += 1
    if g is None:
        return False
    _gripper_stop_best_effort(g)
    # franky.Gripper closes its TCP socket in the C++ destructor.
    try:
        del g
    except Exception:
        pass
    LOG.info("Franka Hand connection released")
    return True


def _clamp_gripper_speed(speed_m_s: float) -> float:
    # Franka Hand continuous speed limit is ~0.1 m/s (libfranka clamps higher values).
    return float(max(0.02, min(0.1, float(speed_m_s))))


def _set_gripper_target(st: AppState, *, width_m: float, speed_m_s: float) -> dict[str, Any]:
    """Record the latest leader width + direction intent; applied by the command loop."""
    _require_gripper(st)
    width = max(0.0, min(float(st.gripper_max_width_m), float(width_m)))
    speed = _clamp_gripper_speed(speed_m_s)
    with st.gripper_lock:
        prev = st.gripper_target_width_m
        if prev is None or abs(width - float(prev)) >= 1e-4:
            if prev is not None:
                st.gripper_target_dir = 1 if width > float(prev) else -1
            st.gripper_target_t_s = time.time()
        st.gripper_target_width_m = width
        st.gripper_target_speed_m_s = speed
        busy = bool(st.gripper_busy)
    return {
        "ok": True,
        "accepted": True,
        "position": width_m_to_bits(width, max_width_m=st.gripper_max_width_m),
        "width_m": float(width),
        "speed_m_s": float(speed),
        "busy": busy,
        "backend": "franka_hand",
    }


def _blocking_gripper_move(st: AppState, *, width_m: float, speed_m_s: float) -> dict[str, Any]:
    """Blocking Move — used when wait=true (open/close one-shots)."""
    g = _require_gripper(st)
    width = max(0.0, min(float(st.gripper_max_width_m), float(width_m)))
    speed = _clamp_gripper_speed(speed_m_s)
    with st.gripper_lock:
        if st.gripper_busy:
            raise HTTPException(status_code=409, detail={"error": "Gripper busy"})
        st.gripper_busy = True
        st.gripper_target_width_m = width
        st.gripper_target_speed_m_s = speed
    try:
        _gripper_stop_best_effort(g)
        ok = bool(g.move(float(width), float(speed)))
        measured = _read_gripper_width_m(g)
        with st.gripper_lock:
            st.gripper_last_cmd_width_m = float(width)
            st.gripper_last_error = None if ok else "move returned False"
        return {
            "ok": bool(ok),
            "accepted": False,
            "position": width_m_to_bits(measured, max_width_m=st.gripper_max_width_m),
            "width_m": float(measured),
            "object_detected": None,
            "backend": "franka_hand",
        }
    except Exception as e:
        with st.gripper_lock:
            st.gripper_last_error = f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=500, detail={"error": str(e)}) from e
    finally:
        with st.gripper_lock:
            st.gripper_busy = False


def _issue_move(st: AppState, g: object, width: float, speed: float, *, preempt: bool) -> None:
    """Start ``g.move`` in a background thread; ``busy`` stays True until it returns.

    ``preempt`` sends ``stop()`` first so the Hand accepts the new Move. A
    generation counter guards against a stale (preempted) thread clearing
    ``busy`` after a newer Move has already started.
    """
    if preempt:
        _gripper_stop_best_effort(g)

    with st.gripper_lock:
        st.gripper_motion_gen += 1
        gen = st.gripper_motion_gen
        st.gripper_busy = True
        st.gripper_goal_width_m = float(width)

    def _run() -> None:
        err: str | None = None
        try:
            ok = bool(g.move(float(width), float(speed)))
            if not ok:
                err = "move returned False"
        except Exception as e:  # CommandException on preempt is expected
            err = f"{type(e).__name__}: {e}"
        finally:
            with st.gripper_lock:
                if st.gripper_motion_gen == gen:
                    st.gripper_busy = False
                    st.gripper_last_cmd_width_m = float(width)
                    st.gripper_last_error = err

    t = threading.Thread(target=_run, name="franka-gripper-move", daemon=True)
    with st.gripper_lock:
        st.gripper_motion_thread = t
    t.start()


def _gripper_command_loop(*, st: AppState) -> None:
    """Franka Hand teleop follower.

    Why chained Moves stutter
    -------------------------
    libfranka ``Gripper.move(width, speed)`` is point-to-point: it ramps up,
    travels, and *decelerates to a stop* at ``width``. It cannot be retargeted
    in flight; a new Move needs ``stop()`` first, which also halts the fingers.
    So any scheme that feeds intermediate leader widths — GELLO's
    wait-for-completion or stop+move per sample — produces one visible halt per
    segment. GELLO gets away with it because its leader is effectively binary.

    Modes (``GRIPPER_FOLLOW_MODE``)
    -------------------------------
    intent (default)
        While the leader is *moving* (target changed within ``settle_s``),
        command one uninterrupted Move toward the extreme in that direction
        (fully open / fully closed). When the leader *settles*, watch the live
        width and ``stop()`` as the fingers pass the settled target
        (``stop_lead_m`` early to absorb latency). Result: one smooth stroke
        that ends where the leader ended; continuous width preserved.
    binary
        Hysteresis open/close (``binary_close_frac`` / ``binary_open_frac``).
        One Move per transition, never a mid-stroke stop. Most robust; what
        most Franka-Hand teleop stacks (incl. GELLO) effectively do.
    waypoint
        Idle-only Move to the latest target (GELLO-style). Stutters on
        continuous input; kept for comparison.
    """
    mode = (st.gripper_follow_mode or "intent").strip().lower()
    period_s = 0.01
    while not st.shutdown.is_set():
        time.sleep(period_s)
        g = st.gripper
        if g is None:
            continue

        with st.gripper_lock:
            if not st.gripper_homed:
                continue
            target = st.gripper_target_width_m
            if target is None:
                continue
            target = float(target)
            speed = _clamp_gripper_speed(st.gripper_target_speed_m_s)
            busy = bool(st.gripper_busy)
            goal = st.gripper_goal_width_m
            max_w = float(st.gripper_max_width_m)
            t_change = float(st.gripper_target_t_s)
            direction = int(st.gripper_target_dir)
            eps = float(st.gripper_width_eps_m)
            deadband = float(st.gripper_deadband_m)
            settle_s = float(st.gripper_settle_s)
            lead = float(st.gripper_stop_lead_m)
            close_frac = float(st.gripper_binary_close_frac)
            open_frac = float(st.gripper_binary_open_frac)
        now = time.time()

        try:
            if mode == "binary":
                frac_closed = 1.0 - (target / max_w if max_w > 0 else 0.0)
                if goal is None:
                    desired = 0.0 if frac_closed >= 0.5 else max_w
                else:
                    goal_is_closed = float(goal) <= max_w * 0.5
                    if goal_is_closed and frac_closed <= open_frac:
                        desired = max_w
                    elif (not goal_is_closed) and frac_closed >= close_frac:
                        desired = 0.0
                    else:
                        desired = float(goal)
                if goal is None or abs(desired - float(goal)) > eps:
                    _issue_move(st, g, desired, speed, preempt=busy)
                continue

            if mode == "waypoint":
                if not busy and (goal is None or abs(target - float(goal)) > deadband):
                    _issue_move(st, g, target, speed, preempt=False)
                continue

            # ── intent ────────────────────────────────────────────────────
            leader_moving = direction != 0 and (now - t_change) < settle_s

            if leader_moving:
                desired = max_w if direction > 0 else 0.0
                if goal is None or abs(desired - float(goal)) > eps:
                    # Already heading the same way → nothing to do (no stop).
                    # Otherwise retarget; stop() only if a Move is in flight.
                    _issue_move(st, g, desired, speed, preempt=busy)
                continue

            # Leader settled at `target`.
            if busy and goal is not None and abs(float(goal) - target) > deadband:
                # Hand is en route to an extreme beyond the settled target:
                # halt exactly as the fingers pass it.
                try:
                    w = _read_gripper_width_m(g)
                except Exception:
                    continue  # can't observe; let the Move finish
                heading_open = float(goal) > target
                reached = (w >= target - lead) if heading_open else (w <= target + lead)
                if reached:
                    _gripper_stop_best_effort(g)
                    with st.gripper_lock:
                        st.gripper_goal_width_m = float(w)
                continue

            if not busy:
                try:
                    w = _read_gripper_width_m(g)
                except Exception:
                    w = float(goal) if goal is not None else target
                if abs(target - w) > deadband:
                    # Small settled correction (e.g. stop() overshoot).
                    _issue_move(st, g, target, speed, preempt=False)
        except Exception as e:
            with st.gripper_lock:
                st.gripper_last_error = f"{type(e).__name__}: {e}"
            LOG.warning("Franka Hand follow loop error: %s", e)


def _home_gripper_gello_style(st: AppState, g: object, *, force: bool) -> dict[str, Any]:
    """Home once (GELLO pattern): stop → homing → settle → measure max width."""
    with st.gripper_lock:
        if st.gripper_homed and not force:
            return {
                "ok": True,
                "is_activated": True,
                "homed": True,
                "skipped": True,
                "max_width_m": st.gripper_max_width_m,
                "backend": "franka_hand",
            }
        if st.gripper_busy:
            raise HTTPException(status_code=409, detail={"error": "Gripper busy"})
        st.gripper_busy = True
        # Pause streaming moves during homing.
        st.gripper_target_width_m = None

    last_exc: Exception | None = None
    ok = False
    try:
        for attempt in (1, 2):
            try:
                _gripper_stop_best_effort(g)
                time.sleep(0.2)
                LOG.info("Franka Hand homing attempt %d/2 ...", attempt)
                if hasattr(g, "homing"):
                    ok = bool(g.homing())
                elif hasattr(g, "open"):
                    ok = bool(g.open(_clamp_gripper_speed(0.1)))
                else:
                    raise RuntimeError("gripper has neither homing() nor open()")
                # GELLO waits after homing before reading max width / accepting commands.
                time.sleep(2.0)
                if not ok:
                    raise RuntimeError("homing returned False")
                break
            except Exception as e:
                last_exc = e
                LOG.warning("Franka Hand homing attempt %d failed: %s", attempt, e)
                _gripper_stop_best_effort(g)
                time.sleep(1.0)
                ok = False
        if not ok:
            raise RuntimeError(f"homing failed: {last_exc}")

        # Prefer device max_width; fall back to measured post-home opening (GELLO).
        max_w = _read_gripper_max_width_m(g)
        try:
            measured = _read_gripper_width_m(g)
            if measured > max_w:
                max_w = measured
            if measured > 0.01:
                max_w = max(max_w, measured)
        except Exception:
            measured = None

        with st.gripper_lock:
            st.gripper_max_width_m = float(max_w)
            st.gripper_homed = True
            st.gripper_last_error = None
            st.gripper_last_cmd_width_m = float(measured) if measured is not None else float(max_w)
            st.gripper_target_width_m = st.gripper_last_cmd_width_m

        LOG.info("Franka Hand homed (max_width=%.4f m)", st.gripper_max_width_m)
        return {
            "ok": True,
            "is_activated": True,
            "homed": True,
            "skipped": False,
            "max_width_m": st.gripper_max_width_m,
            "width_m": measured,
            "backend": "franka_hand",
        }
    except Exception as e:
        with st.gripper_lock:
            st.gripper_homed = False
            st.gripper_last_error = f"{type(e).__name__}: {e}"
        raise
    finally:
        with st.gripper_lock:
            st.gripper_busy = False


def _try_recover_from_errors(*, st: AppState, context: str, exc: Exception | None = None) -> None:
    """
    Best-effort error recovery hook.

    Franky can enter an error state after motion/preemption/comm faults; attempt to recover
    whenever an exception occurs so subsequent commands have a chance to work.
    """
    robot = st.robot
    if robot is None:
        return
    try:
        robot.recover_from_errors()
    except Exception as e:
        # Keep the original error as primary; append recovery failure details.
        base = f"{type(exc).__name__}: {exc}" if exc is not None else None
        extra = f"recover_from_errors failed ({context}): {type(e).__name__}: {e}"
        with st.lock:
            st.last_control_error = (f"{base} | {extra}" if base else extra)


def _control_loop(*, st: AppState, control_hz: int, timeout_s: float) -> None:
    """Apply latest target at a fixed rate using Franky JointMotion(JointState(...))."""
    period_s = 1.0 / float(control_hz)

    robot = st.robot
    assert robot is not None
    robot.recover_from_errors()

    while not st.shutdown.is_set():
        time.sleep(period_s)

        with st.lock:
            latest = st.latest_target
            stop_latched = st.stop_latched
            timeout_s = st.command_timeout_s

        if stop_latched or latest is None:
            continue

        age_s = time.time() - latest.accepted_timestamp_s
        if age_s > timeout_s:
            try:
                robot.move(franky.JointStopMotion())
            except Exception as e:
                st.last_control_error = f"{type(e).__name__}: {e}"
                _try_recover_from_errors(st=st, context="timeout_stop", exc=e)
            finally:
                with st.lock:
                    st.stop_latched = True
                    st.latest_target = None
            continue

        try:
            target = franky.JointState(latest.positions, latest.velocities)
            # Keep holding the last target until preempted (better streaming behavior).
            motion = franky.JointMotion(target)
            robot.move(motion, asynchronous=True)
            st.last_control_error = None
        except Exception as e:
            st.last_control_error = f"{type(e).__name__}: {e}"
            _try_recover_from_errors(st=st, context="move_joint_motion", exc=e)
            with st.lock:
                st.stop_latched = True
                st.latest_target = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialize robot and start control thread."""
    logging.basicConfig(level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL))

    robot_ip = os.getenv("FRANKY_ROBOT_IP", DEFAULT_FRANKY_ROBOT_IP)
    control_hz = _get_env_int("CONTROL_HZ", DEFAULT_CONTROL_HZ)
    timeout_s = _get_env_float("COMMAND_TIMEOUT_S", DEFAULT_COMMAND_TIMEOUT_S)

    st = AppState(
        robot=None,
        lock=threading.Lock(),
        latest_target=None,
        stop_latched=True,  # start latched until first target is posted
        shutdown=threading.Event(),
        control_thread=None,
        last_control_error=None,
        command_timeout_s=timeout_s,
    )
    app.state.franky_state = st
    st.robot = franky.Robot(robot_ip)
    st.robot.relative_dynamics_factor = franky.RelativeDynamicsFactor(0.4, 0.3, 0.1)

    # Franka Hand: do NOT connect at startup. Holding a libfranka Gripper
    # connection makes Desk show "End effector not connected" (red) for as
    # long as the connection is open. Connect lazily on POST /connect and
    # release on POST /disconnect (teleop does both).
    if _gripper_enabled():
        st.gripper_cmd_thread = threading.Thread(
            target=_gripper_command_loop,
            kwargs={"st": st},
            name="franka-gripper-cmd",
            daemon=True,
        )
        st.gripper_cmd_thread.start()
        # Follow-tuning knobs (see _gripper_command_loop docstring).
        st.gripper_follow_mode = os.getenv("GRIPPER_FOLLOW_MODE", st.gripper_follow_mode).strip().lower()
        st.gripper_settle_s = _get_env_float("GRIPPER_SETTLE_S", st.gripper_settle_s)
        st.gripper_deadband_m = _get_env_float("GRIPPER_DEADBAND_M", st.gripper_deadband_m)
        st.gripper_stop_lead_m = _get_env_float("GRIPPER_STOP_LEAD_M", st.gripper_stop_lead_m)
        st.gripper_binary_close_frac = _get_env_float("GRIPPER_BINARY_CLOSE_FRAC", st.gripper_binary_close_frac)
        st.gripper_binary_open_frac = _get_env_float("GRIPPER_BINARY_OPEN_FRAC", st.gripper_binary_open_frac)
        LOG.info(
            "Franka Hand enabled (lazy connect via POST /connect); follow_mode=%s settle_s=%.3f "
            "deadband_m=%.4f stop_lead_m=%.4f",
            st.gripper_follow_mode, st.gripper_settle_s, st.gripper_deadband_m, st.gripper_stop_lead_m,
        )
    else:
        LOG.info("Franka Hand disabled (FRANKA_GRIPPER=0)")

    # Start background control thread (single uvicorn worker).
    if st.robot is not None:
        st.control_thread = threading.Thread(
            target=_control_loop,
            kwargs={"st": st, "control_hz": control_hz, "timeout_s": timeout_s},
            daemon=True,
        )
        st.control_thread.start()

    try:
        yield
    finally:
        st.shutdown.set()
        if st.control_thread is not None:
            st.control_thread.join(timeout=2.0)
        if st.gripper_cmd_thread is not None:
            st.gripper_cmd_thread.join(timeout=2.0)
        _release_gripper(st)


app = FastAPI(lifespan=_lifespan)


# @app.middleware("http")
# async def _optional_ip_allowlist(request: Request, call_next):
#     """Optionally block requests unless ALLOWED_CLIENT_IP matches the peer IP."""
#     allowed = os.getenv("ALLOWED_CLIENT_IP", DEFAULT_ALLOWED_CLIENT_IP)
#     if allowed:
#         client_ip = request.client.host if request.client else None
#         if client_ip not in (allowed, "127.0.0.1", "::1"):
#             return JSONResponse(status_code=403, content={"detail": "forbidden"})
#     return await call_next(request)


def _get_app_state(request: Request) -> AppState:
    """Get app state."""
    return request.app.state.franky_state


@app.get("/")
def root():
    """Landing page (HTML) showing robot/service status."""
    robot_ip_json = JSONResponse(content=os.getenv("FRANKY_ROBOT_IP", DEFAULT_FRANKY_ROBOT_IP)).body.decode("utf-8")
    html = LANDING_PAGE_HTML % {"robot_ip_json": robot_ip_json}
    return HTMLResponse(content=html)


@app.get("/health")
def health(request: Request):
    """Health/status endpoint."""
    current_app_state = _get_app_state(request)
    with current_app_state.gripper_lock:
        gripper_connected = current_app_state.gripper is not None
        gripper_busy = bool(current_app_state.gripper_busy)
        gripper_homed = bool(current_app_state.gripper_homed)
        gripper_err = current_app_state.gripper_last_error
    return {
        "robot_connected": current_app_state.robot is not None,
        "stop_latched": current_app_state.stop_latched,
        "has_target": current_app_state.latest_target is not None,
        "last_control_error": current_app_state.last_control_error,
        "gripper_connected": gripper_connected,
        "gripper_homed": gripper_homed,
        "gripper_busy": gripper_busy,
        "gripper_last_error": gripper_err,
    }


@app.post("/target_joint_state")
def post_target_joint_state(payload: JointTargetIn, request: Request) -> JointTargetOut:
    """Set the latest joint target (positions/velocities)."""
    _validate_vec("positions", payload.positions)
    _validate_vec("velocities", payload.velocities)

    current_app_state = _get_app_state(request)
    _require_robot(current_app_state)

    now = time.time()
    target = LatestTarget(
        positions=[float(x) for x in payload.positions],
        velocities=[float(x) for x in payload.velocities],
        seq=payload.seq,
        accepted_timestamp_s=now,
    )
    with current_app_state.lock:
        current_app_state.latest_target = target
        current_app_state.stop_latched = False

    return JointTargetOut(
        positions=target.positions,
        velocities=target.velocities,
        seq=target.seq,
        accepted_timestamp_s=target.accepted_timestamp_s,
        age_s=0.0,
        stop_latched=False,
    )


@app.get("/target_joint_state")
def get_target_joint_state(request: Request) -> JointTargetOut:
    """Get the latest accepted target and its age."""
    current_app_state = _get_app_state(request)
    with current_app_state.lock:
        latest = current_app_state.latest_target
        stop_latched = current_app_state.stop_latched

    if latest is None:
        return JointTargetOut(
            positions=[0.0] * N_JOINTS,
            velocities=[0.0] * N_JOINTS,
            seq=None,
            accepted_timestamp_s=0.0,
            age_s=EFFECTIVE_FLOAT_INF_SECONDS,
            stop_latched=stop_latched,
        )

    age_s = time.time() - latest.accepted_timestamp_s
    return JointTargetOut(
        positions=latest.positions,
        velocities=latest.velocities,
        seq=latest.seq,
        accepted_timestamp_s=latest.accepted_timestamp_s,
        age_s=age_s,
        stop_latched=stop_latched,
    )


@app.post("/stop")
def stop(request: Request) -> StopOut:
    """Stop the robot (joint position control mode) and latch stop."""
    current_app_state = _get_app_state(request)
    now = time.time()

    with current_app_state.lock:
        current_app_state.stop_latched = True
        current_app_state.latest_target = None

    if current_app_state.robot is not None:
        try:
            current_app_state.robot.move(franky.JointStopMotion())
        except Exception as e:
            current_app_state.last_control_error = f"{type(e).__name__}: {e}"
            _try_recover_from_errors(st=current_app_state, context="stop_endpoint", exc=e)

    return StopOut(stopped=True, timestamp_s=now)


@app.post("/go_home")
def go_home(request: Request) -> JointTargetOut:
    """
    Convenience endpoint for the landing page: set target joint state to HOME_POSITION.
    This clears the stop latch and lets the background control thread drive the robot.
    """
    current_app_state = _get_app_state(request)
    _require_robot(current_app_state)

    now = time.time()
    target = LatestTarget(
        positions=[float(x) for x in HOME_POSITION],
        velocities=[0.0] * N_JOINTS,
        seq=None,
        accepted_timestamp_s=now,
    )
    with current_app_state.lock:
        current_app_state.latest_target = target
        current_app_state.stop_latched = False

    return JointTargetOut(
        positions=target.positions,
        velocities=target.velocities,
        seq=target.seq,
        accepted_timestamp_s=target.accepted_timestamp_s,
        age_s=0.0,
        stop_latched=False,
    )


def _to_float_list(x: Any) -> list[float]:
    # Handle NumPy arrays directly
    if isinstance(x, np.ndarray):
        return [float(v) for v in x.tolist()]

    # Handle other iterable containers (lists, tuples, etc.)
    try:
        # Strings/bytes should not be treated as sequences of numbers
        if isinstance(x, (str, bytes)):
            raise TypeError
        return [float(v) for v in x]
    except TypeError:
        # Fallback: treat as a single scalar
        return [float(x)]


@app.get("/joint_state")
def get_joint_state(request: Request):
    st = _get_app_state(request)
    robot = _require_robot(st)

    js = getattr(robot, "current_joint_state")
    positions = _to_float_list(js.position)
    velocities = _to_float_list(js.velocity)
    return {"positions": positions, "velocities": velocities}


@app.get("/urdf")
def get_urdf(request: Request):
    """Return URDF if exposed by the robot wrapper."""
    st = _get_app_state(request)
    robot = _require_robot(st)

    # Upstream docs show robot.model_urdf; also keep robot.urdf compatibility if present.
    if hasattr(robot, "model_urdf"):
        return {"urdf": robot.model_urdf}
    if hasattr(robot, "urdf"):
        return {"urdf": robot.urdf}
    raise HTTPException(status_code=404, detail="urdf not available on this franky build")


@app.get("/command_timeout")
def get_command_timeout(request: Request) -> CommandTimeoutOut:
    """Get current command timeout (seconds)."""
    current_app_state = _get_app_state(request)
    with current_app_state.lock:
        return CommandTimeoutOut(command_timeout_s=current_app_state.command_timeout_s)


@app.post("/command_timeout")
def set_command_timeout(payload: CommandTimeoutIn, request: Request) -> CommandTimeoutOut:
    """Set command timeout (seconds)."""
    current_app_state = _get_app_state(request)
    with current_app_state.lock:
        current_app_state.command_timeout_s = float(payload.command_timeout_s)
        return CommandTimeoutOut(command_timeout_s=current_app_state.command_timeout_s)


# ── Franka Hand (Robotiq-compatible API for GripperClient / teleop) ───────────

@app.post("/connect")
def gripper_connect(request: Request) -> dict[str, Any]:
    """Open the libfranka Gripper connection (lazy).

    While this connection is held, Desk shows the Hand as "not connected";
    call POST /disconnect when done to give the card back to Desk.
    """
    st = _get_app_state(request)
    if not _gripper_enabled():
        raise HTTPException(status_code=503, detail={"error": "Franka Hand disabled (FRANKA_GRIPPER=0)"})
    if st.gripper is None:
        robot_ip = os.getenv("FRANKY_ROBOT_IP", DEFAULT_FRANKY_ROBOT_IP)
        try:
            g = franky.Gripper(robot_ip)
            max_w = _read_gripper_max_width_m(g)
            with st.gripper_lock:
                st.gripper = g
                st.gripper_max_width_m = max_w
                st.gripper_last_error = None
                st.gripper_homed = False
            LOG.info("Franka Hand connected (max_width=%.3f m)", max_w)
        except Exception as e:
            with st.gripper_lock:
                st.gripper_last_error = f"{type(e).__name__}: {e}"
                st.gripper = None
            LOG.warning("Franka Hand connect failed: %s", st.gripper_last_error)
    return {
        "connected": st.gripper is not None,
        "last_connect_error": st.gripper_last_error,
        "max_width_m": st.gripper_max_width_m if st.gripper is not None else None,
        "backend": "franka_hand",
    }


@app.post("/disconnect")
def gripper_disconnect(request: Request) -> dict[str, Any]:
    """Release the libfranka Gripper connection so Desk shows the Hand again."""
    st = _get_app_state(request)
    released = _release_gripper(st)
    return {"ok": True, "released": released, "connected": False, "backend": "franka_hand"}


@app.post("/activate")
def gripper_activate(
    request: Request,
    wait: bool = Query(True),
    home: bool = Query(True, description="Run Franka Hand homing (slow). Set false to skip."),
    force: bool = Query(False, description="Re-home even if already homed."),
) -> dict[str, Any]:
    """Home the Franka Hand once (GELLO-style). Skips if already homed unless force=true."""
    st = _get_app_state(request)
    g = _require_gripper(st)

    if not home:
        with st.gripper_lock:
            st.gripper_homed = True
        return {
            "ok": True,
            "is_activated": True,
            "homed": True,
            "skipped": True,
            "max_width_m": st.gripper_max_width_m,
            "backend": "franka_hand",
        }

    def _run_home() -> dict[str, Any]:
        return _home_gripper_gello_style(st, g, force=force)

    if not wait:
        def _bg() -> None:
            try:
                _run_home()
            except Exception as e:
                LOG.warning("async activate failed: %s", e)

        threading.Thread(target=_bg, name="franka-gripper-activate", daemon=True).start()
        return {
            "ok": True,
            "accepted": True,
            "is_activated": None,
            "homed": None,
            "busy": True,
            "backend": "franka_hand",
        }

    try:
        return _run_home()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)}) from e


@app.post("/reset")
def gripper_reset(request: Request) -> dict[str, Any]:
    """Best-effort stop; clears homed flag so next activate will re-home."""
    st = _get_app_state(request)
    g = _require_gripper(st)
    _gripper_stop_best_effort(g)
    with st.gripper_lock:
        st.gripper_busy = False
        st.gripper_homed = False
        st.gripper_target_width_m = None
        st.gripper_last_cmd_width_m = None
    return {"ok": True, "backend": "franka_hand"}


@app.post("/reset_activate")
def gripper_reset_activate(request: Request) -> dict[str, Any]:
    gripper_reset(request)
    return gripper_activate(request, wait=True, home=True, force=True)


@app.post("/open")
def gripper_open(
    request: Request,
    req: GripperGoToBitsIn | None = None,
    wait: bool = Query(True),
) -> dict[str, Any]:
    st = _get_app_state(request)
    if req is None:
        req = GripperGoToBitsIn(position=0)
    width = bits_to_width_m(0, max_width_m=st.gripper_max_width_m)
    speed = bits_to_speed_m_s(req.speed)
    if not wait:
        return _set_gripper_target(st, width_m=width, speed_m_s=speed)
    return _blocking_gripper_move(st, width_m=width, speed_m_s=speed)


@app.post("/close")
def gripper_close(
    request: Request,
    req: GripperGoToBitsIn | None = None,
    wait: bool = Query(True),
) -> dict[str, Any]:
    """Close via Move to width=0 (no Grasp — Grasp trips Desk errors in teleop)."""
    st = _get_app_state(request)
    if req is None:
        req = GripperGoToBitsIn(position=255)
    speed = bits_to_speed_m_s(req.speed)
    if not wait:
        return _set_gripper_target(st, width_m=0.0, speed_m_s=speed)
    return _blocking_gripper_move(st, width_m=0.0, speed_m_s=speed)


@app.post("/go_to")
def gripper_go_to(
    request: Request,
    req: GripperGoToBitsIn,
    wait: bool = Query(True),
) -> dict[str, Any]:
    """Move to Robotiq-style position bits (0=open … 255=closed). Move only, no Grasp."""
    st = _get_app_state(request)
    if not st.gripper_homed:
        raise HTTPException(
            status_code=409,
            detail={"error": "Gripper not homed", "hint": "POST /activate first"},
        )
    width = bits_to_width_m(req.position, max_width_m=st.gripper_max_width_m)
    speed = bits_to_speed_m_s(req.speed)
    if not wait:
        return _set_gripper_target(st, width_m=width, speed_m_s=speed)
    return _blocking_gripper_move(st, width_m=width, speed_m_s=speed)


@app.get("/gripper_state")
def gripper_state(
    request: Request,
    closed_threshold: int = Query(128, ge=0, le=255),
) -> dict[str, Any]:
    """Robotiq-compatible state for GripperClient / teleop observation."""
    st = _get_app_state(request)
    g = _require_gripper(st)
    with st.gripper_lock:
        busy = bool(st.gripper_busy)
        homed = bool(st.gripper_homed)
        max_w = float(st.gripper_max_width_m)

    # Width is read over the gripper's UDP state stream, independent of the TCP
    # command channel, so it is safe to sample while a Move is in flight. This
    # keeps recorded gripper observations live during motion.
    try:
        width = _read_gripper_width_m(g)
    except Exception as e:
        if busy:
            return {
                "connected": True,
                "position_bits": None,
                "max_position_bits": 255,
                "position_frac": None,
                "width_m": None,
                "max_width_m": max_w,
                "is_closed": None,
                "is_open": None,
                "is_activated": homed,
                "is_calibrated": homed,
                "busy": True,
                "backend": "franka_hand",
            }
        raise HTTPException(
            status_code=503,
            detail={"error": "Failed to read Franka Hand width", "exc": repr(e)},
        ) from e

    pos = width_m_to_bits(width, max_width_m=max_w)
    is_closed = bool(pos >= int(closed_threshold))
    return {
        "connected": True,
        "position_bits": pos,
        "max_position_bits": 255,
        "position_frac": float(pos) / 255.0,
        "width_m": float(width),
        "max_width_m": max_w,
        "is_closed": is_closed,
        "is_open": not is_closed,
        "is_activated": homed,
        "is_calibrated": homed,
        "busy": busy,
        "backend": "franka_hand",
    }


@app.get("/position")
def gripper_position(request: Request) -> dict[str, Any]:
    st = gripper_state(request)
    return {"position": st.get("position_bits"), "busy": st.get("busy"), "backend": "franka_hand"}


@app.get("/is_activated")
def gripper_is_activated(request: Request) -> dict[str, Any]:
    st = _get_app_state(request)
    _require_gripper(st)
    with st.gripper_lock:
        return {"is_activated": bool(st.gripper_homed), "busy": bool(st.gripper_busy)}


@app.get("/is_calibrated")
def gripper_is_calibrated(request: Request) -> dict[str, Any]:
    return gripper_is_activated(request)


def main():
    """Entry point for franky-service CLI."""
    uvicorn.run("droid_plus.services.franky_service:app", host="0.0.0.0", port=DEFAULT_SERVICE_PORT, workers=1)


if __name__ == "__main__":
    main()
