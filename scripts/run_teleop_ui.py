#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# sorting--
"""
Web UI teleop runner — browser interface for SO-101 teleop episodes.

Parallels ``scripts/run_experiment_ui.py`` but for data-generation. The
SO-101 leader arm is connected at launch; each episode is started/stopped
from the browser (Enter / Esc, or the terminal's ESC key).

Usage:
    python scripts/run_teleop_ui.py
    python scripts/run_teleop_ui.py --port /dev/ttyACM1 --web-port 54325
    python scripts/run_teleop_ui.py --gripper franka \\
        --left-camera-id 31262010 --wrist-camera-id 14633207 \\
        --output-dir output_100926
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from droid_plus.analysis.end_effector_pose import compute_and_save_ee_trajectory_single
from droid_plus.constants import (
    FRANKY_SERVICE_URL,
    GRIPPER_SERVICE_URL,
    LEFT_CAMERA_SERIAL,
    RECORD_JPEG_QUALITY,
    RIGHT_CAMERA_SERIAL,
    WRIST_CAMERA_SERIAL,
)
from droid_plus.datagen import (
    DEFAULT_MIN_EE_Z,
    TeleopSessionConfig,
    build_fk_model,
    connect_so101,
    finalize_teleop_episode_recording,
    init_gripper,
    make_teleop_run_dir,
    run_teleop_episode,
)
from droid_plus.eval.episode_runner import EpisodeConfig, EpisodeResult
from droid_plus.eval.experiment_setup import wait_for_cameras
from droid_plus.logging import EpisodeRecorder
from droid_plus.logging.phase_labels import PhaseSelector
from droid_plus.robot import DroidPlus
from droid_plus.utils import KeyPoller

DEFAULT_PORT = 54325


# ── Shared state ─────────────────────────────────────────────────────────────

@dataclass
class EpisodeHistory:
    episode_idx: int
    instruction: str
    steps: int
    duration_s: float
    episode_dir: str | None = None
    experiment_name: str = ""
    task_name: str = ""
    labels: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppState:
    droid: DroidPlus
    teleop: Any
    gripper_initialized: bool
    pin_model: Any
    pin_data: Any
    ee_frame: str
    session: TeleopSessionConfig
    so101_port: str
    base_run_dir: str | None
    # Episode control
    episode_idx: int = 0
    episode_thread: threading.Thread | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    current_result: EpisodeResult | None = None
    current_config: EpisodeConfig | None = None
    status: str = "idle"  # "idle" | "running"
    phases: PhaseSelector = field(default_factory=PhaseSelector)
    t_start: float = 0.0
    history: list[EpisodeHistory] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _run_episode_thread(state: AppState) -> None:
    """Background thread that runs a single teleop episode."""
    try:
        recorder: EpisodeRecorder | None = None
        if state.session.record and state.base_run_dir is not None:
            record_cameras = [
                name for name, cid in (
                    ("left", state.droid.left_camera_id),
                    ("wrist", state.droid.wrist_camera_id),
                    ("right", state.droid.right_camera_id),
                )
                if cid
            ]
            recorder = EpisodeRecorder(
                base_run_dir=state.base_run_dir,
                episode_idx=state.episode_idx,
                jpeg_quality=state.session.record_jpeg_quality,
                cameras=record_cameras or ["left", "wrist"],
            )

        result = run_teleop_episode(
            config=state.current_config,
            session=state.session,
            droid=state.droid,
            teleop=state.teleop,
            gripper_initialized=state.gripper_initialized,
            pin_model=state.pin_model,
            pin_data=state.pin_data,
            ee_frame=state.ee_frame,
            recorder=recorder,
            should_stop=state.stop_event.is_set,
            get_phase=state.phases.get,
        )

        with state.lock:
            state.current_result = result

        if state.session.record and result.recorder is not None:
            finalize_teleop_episode_recording(
                result=result,
                config=state.current_config,
                session=state.session,
            )
            try:
                compute_and_save_ee_trajectory_single(
                    episode_dir=result.recorder.episode_dir,
                    overwrite=False,
                    verbose=True,
                )
            except Exception as e:
                print(f"Warning: Failed to compute EE trajectory: {e}")

        duration = result.t_end - result.t_start
        ep_dir = result.recorder.episode_dir if result.recorder else None
        with state.lock:
            state.history.insert(0, EpisodeHistory(
                episode_idx=state.episode_idx,
                instruction=state.current_config.instruction,
                steps=result.seq,
                duration_s=duration,
                episode_dir=ep_dir,
                experiment_name=state.current_config.experiment or "",
                task_name=state.current_config.task or "",
            ))
            state.episode_idx += 1

    except Exception as e:
        print(f"Episode thread error: {e}")
    finally:
        with state.lock:
            state.status = "idle"
            state.episode_thread = None


# ── Landing page HTML ────────────────────────────────────────────────────────

LANDING_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>teleop_service</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: #0b0f14; color: #e6edf3; display: flex; flex-direction: column; min-height: 100vh; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; }

    .header { display: flex; align-items: center; gap: 14px; padding: 16px 24px; border-bottom: 1px solid #223042; flex-wrap: wrap; }
    .header-title { font-size: 18px; font-weight: 800; }
    .muted { color: #9fb2c8; }

    .badge { display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 13px; font-weight: 700; letter-spacing: 0.3px; }
    .badge-idle { background: #0d2818; color: #3fb950; border: 1px solid #1a4028; }
    .badge-running { background: #2d2000; color: #d29922; border: 1px solid #4a3500; }

    .card { background: #0f1620; border: 1px solid #223042; border-radius: 10px; padding: 20px; width: 100%; max-width: 720px; }
    .card-label { font-size: 15px; font-weight: 700; margin-bottom: 12px; }

    .main { flex: 1; display: flex; flex-direction: column; align-items: center; padding: 24px; gap: 16px; }

    .field-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 20px; }
    .field-cell label { display: block; font-size: 12px; color: #9fb2c8; margin-bottom: 4px; }
    .field-cell .hint { font-size: 11px; color: #6b7f97; font-weight: 400; margin-left: 6px; }
    .field-input { width: 100%; box-sizing: border-box; background: #0b111a; color: #e6edf3; border: 1px solid #223042; border-radius: 6px; padding: 6px 10px; font-size: 13px; font-family: inherit; outline: none; transition: border-color 0.15s; }
    .field-input:focus { border-color: #1f6feb; }
    .field-input:disabled { opacity: 0.5; cursor: not-allowed; }
    .field-readonly { background: #0b111a; color: #9fb2c8; border: 1px solid #192434; border-radius: 6px; padding: 6px 10px; font-size: 13px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }

    .instruction-label { font-size: 16px; font-weight: 700; margin-bottom: 10px; }
    .instruction-input { width: 100%; box-sizing: border-box; background: #0b111a; color: #e6edf3; border: 1px solid #223042; border-radius: 8px; padding: 18px 20px; font-size: 20px; line-height: 1.4; font-family: inherit; resize: vertical; min-height: 110px; outline: none; transition: border-color 0.15s; }
    .instruction-input:focus { border-color: #1f6feb; }
    .instruction-input:disabled { opacity: 0.5; cursor: not-allowed; }
    .instruction-input::placeholder { color: #4a5568; }
    .instruction-row { display: flex; align-items: start; gap: 10px; }
    .instruction-row .instruction-input { flex: 1; }
    .clear-btn { background: #223042; color: #9fb2c8; border: 1px solid #2b3b52; border-radius: 8px; padding: 12px 14px; font-size: 14px; cursor: pointer; white-space: nowrap; transition: background 0.15s, color 0.15s; }
    .clear-btn:hover { background: #2b3b52; color: #e6edf3; }
    .instruction-hints { display: flex; justify-content: space-between; margin-top: 8px; margin-bottom: 28px; font-size: 13px; color: #9fb2c8; }
    .instruction-hints kbd { background: #223042; border: 1px solid #2b3b52; border-radius: 4px; padding: 1px 6px; font-family: inherit; font-size: 12px; }

    .status-area { width: 100%; max-width: 720px; min-height: 40px; }
    .status-running { background: #0f1620; border: 1px solid #4a3500; border-radius: 10px; padding: 14px 18px; }
    .status-text { font-size: 15px; font-weight: 600; color: #d29922; }
    .status-detail { font-size: 13px; color: #9fb2c8; margin-top: 4px; }

    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
    .pulse { animation: pulse 1.5s ease-in-out infinite; }

    .history-title { font-size: 15px; font-weight: 700; margin-bottom: 10px; }
    .history-empty { color: #9fb2c8; font-size: 13px; }
    .history-item { padding: 6px 0; border-bottom: 1px solid #192434; font-size: 13px; }
    .history-item:last-child { border-bottom: none; }
    .history-row { display: flex; justify-content: space-between; align-items: baseline; cursor: pointer; }
    .history-row:hover { color: #58a6ff; }
    .history-idx { color: #9fb2c8; min-width: 30px; }
    .history-instruction { flex: 1; margin: 0 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .history-stats { color: #9fb2c8; white-space: nowrap; }
    .history-tag { display: inline-block; background: #223042; border-radius: 4px; padding: 1px 6px; font-size: 11px; color: #9fb2c8; margin-left: 6px; }
    .history-tag.exp { background: #1a2b40; color: #58a6ff; }
    .history-tag.task { background: #1a3a28; color: #7ee787; }
    .history-labels { font-size: 11px; color: #9fb2c8; margin-top: 2px; padding-left: 30px; }
    .label-form { display: none; margin-top: 8px; padding: 10px 12px; background: #0b111a; border: 1px solid #223042; border-radius: 8px; }
    .label-form.open { display: block; }
    .label-row { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; flex-wrap: wrap; }
    .label-row:last-child { margin-bottom: 0; }
    .label-field { font-size: 12px; color: #9fb2c8; }
    .label-field label { margin-right: 4px; }
    .label-toggle { display: inline-flex; gap: 2px; }
    .label-toggle button { background: #192434; color: #9fb2c8; border: 1px solid #223042; border-radius: 4px; padding: 2px 10px; font-size: 12px; cursor: pointer; }
    .label-toggle button.active-yes { background: #0d2818; color: #3fb950; border-color: #1a4028; }
    .label-toggle button.active-no { background: #2d0a0a; color: #f85149; border-color: #4a1414; }
    .label-toggle button:hover { border-color: #1f6feb; }
    .label-input { background: #0b111a; color: #e6edf3; border: 1px solid #223042; border-radius: 4px; padding: 3px 8px; font-size: 12px; width: 60px; }
    .label-input-wide { width: 180px; }
    .label-save { background: #1f6feb; color: #fff; border: none; border-radius: 4px; padding: 4px 14px; font-size: 12px; font-weight: 600; cursor: pointer; }
    .label-save:hover { background: #388bfd; }
    .label-saved { color: #3fb950; font-size: 12px; margin-left: 8px; }
  </style>
</head>
<body>
  <div class="header">
    <div class="header-title">teleop_service</div>
    <span id="status_badge" class="badge badge-idle">idle</span>
    <div class="muted" style="margin-left: auto;">Episode <code id="episode_idx">0</code></div>
  </div>

  <div class="main">
    <div class="card">
      <div class="card-label">Teleop</div>
      <div class="field-grid">
        <div class="field-cell">
          <label>SO-101 port <span class="hint">set at launch with <code>--port</code></span></label>
          <div class="field-readonly" id="so101_port">—</div>
        </div>
        <div class="field-cell">
          <label>Gripper</label>
          <div class="field-readonly" id="gripper_status">—</div>
        </div>
        <div class="field-cell">
          <label>Control rate (Hz) <span class="hint"><code>--rate-hz</code></span></label>
          <div class="field-readonly" id="rate_hz">—</div>
        </div>
        <div class="field-cell">
          <label>Record rate (Hz) <span class="hint"><code>--record-rate-hz</code></span></label>
          <div class="field-readonly" id="record_rate_hz">—</div>
        </div>
        <div class="field-cell">
          <label>Min EE z (m) <span class="hint">table-safety floor</span></label>
          <input type="number" id="min_z" class="field-input" step="0.01" />
        </div>
        <div class="field-cell">
          <label>Recording</label>
          <div class="field-readonly" id="record_status">—</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-label">Episode</div>
      <div class="instruction-row">
        <textarea id="instruction" class="instruction-input" rows="3"
          placeholder="Optional: description of what you're demonstrating..." autofocus></textarea>
        <button class="clear-btn" onclick="instructionEl.value=''; instructionEl.focus();">Clear</button>
      </div>
      <div class="instruction-hints">
        <span><kbd>Enter</kbd> to start episode</span>
        <span><kbd>Esc</kbd> to stop episode</span>
      </div>
      <div class="field-grid">
        <div class="field-cell">
          <label>Experiment name <span class="hint">Unique name identifying this experiment.</span></label>
          <input type="text" id="experiment_name" class="field-input" placeholder="e.g. banana_in_bowl_v3" />
        </div>
        <div class="field-cell">
          <label>Task name <span class="hint">Consistent across variations of the same task.</span></label>
          <input type="text" id="task_name" class="field-input" placeholder="e.g. banana_in_bowl" />
        </div>
        <div class="field-cell">
          <label>Max walltime (s) <span class="hint">Blank = unlimited.</span></label>
          <input type="number" id="max_walltime" class="field-input" placeholder="unlimited" step="0.1" />
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-label">Phase: <span id="current_phase">0</span></div>
      <p>Starts at 0 each episode. Pause ~0.5 s at a boundary, select the phase, then continue.
      Number keys 0–9 work when focus is outside a text field.</p>
      <div id="phase_buttons"></div>
      <span id="phase_error" role="alert"></span>
    </div>
    <div class="status-area" id="status_area"></div>

    <div class="card">
      <div class="history-title">History</div>
      <div id="history_list"><div class="history-empty">No episodes yet.</div></div>
    </div>
  </div>

  <script>
    const instructionEl = document.getElementById("instruction");
    const experimentEl = document.getElementById("experiment_name");
    const taskEl = document.getElementById("task_name");
    const walltimeEl = document.getElementById("max_walltime");
    const minZEl = document.getElementById("min_z");
    const badgeEl = document.getElementById("status_badge");
    const episodeIdxEl = document.getElementById("episode_idx");
    const statusArea = document.getElementById("status_area");
    const historyList = document.getElementById("history_list");
    const so101PortEl = document.getElementById("so101_port");
    const gripperStatusEl = document.getElementById("gripper_status");
    const rateHzEl = document.getElementById("rate_hz");
    const recordRateHzEl = document.getElementById("record_rate_hz");
    const recordStatusEl = document.getElementById("record_status");

    let currentStatus = "idle";
    let configRateHz = null;
    let lastHistoryLen = 0;
    const pendingLabels = {};

    // ── Restore persisted fields ──
    const LS_KEY = "teleop_service_state_v1";
    function loadPersisted() {
      try {
        const raw = localStorage.getItem(LS_KEY);
        if (!raw) return;
        const s = JSON.parse(raw);
        if (s.experiment) experimentEl.value = s.experiment;
        if (s.task) taskEl.value = s.task;
        if (s.walltime != null) walltimeEl.value = s.walltime;
      } catch(e) {}
    }
    function savePersisted() {
      try {
        localStorage.setItem(LS_KEY, JSON.stringify({
          experiment: experimentEl.value,
          task: taskEl.value,
          walltime: walltimeEl.value,
        }));
      } catch(e) {}
    }
    [experimentEl, taskEl, walltimeEl].forEach(el => {
      el.addEventListener("input", savePersisted);
    });
    loadPersisted();

    async function loadConfig() {
      try {
        const r = await fetch("/config");
        const d = await r.json();
        configRateHz = d.rate_hz;
        so101PortEl.textContent = d.so101_port || "—";
        gripperStatusEl.textContent = d.gripper_initialized ? "initialized" : "disabled";
        rateHzEl.textContent = d.rate_hz;
        recordRateHzEl.textContent = d.record_rate_hz;
        recordStatusEl.textContent = d.record ? `to ${d.base_run_dir}` : "disabled";
        if (minZEl.value === "") minZEl.value = d.min_z;
      } catch(e) { console.error("Failed to load config", e); }
    }
    loadConfig();

    minZEl.addEventListener("change", async () => {
      const v = parseFloat(minZEl.value);
      if (!Number.isFinite(v)) return;
      try { await fetch("/min_z", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({min_z: v}),
      }); } catch(e) {}
    });

    // ── Poll status ──
    setInterval(async () => {
      try {
        const r = await fetch("/status");
        const s = await r.json();
        updateUI(s);
      } catch(e) {}
    }, 500);

    function updateUI(s) {
      currentStatus = s.status;
      document.getElementById("current_phase").textContent = s.phase_index;
      document.querySelectorAll("#phase_buttons button").forEach(b => b.disabled = s.status !== "running");
      episodeIdxEl.textContent = s.episode_idx;

      if (s.status === "running") {
        badgeEl.className = "badge badge-running pulse";
        badgeEl.textContent = "running";
      } else {
        badgeEl.className = "badge badge-idle";
        badgeEl.textContent = "idle";
      }

      const disabled = (s.status === "running");
      instructionEl.disabled = disabled;
      experimentEl.disabled = disabled;
      taskEl.disabled = disabled;
      walltimeEl.disabled = disabled;
      minZEl.disabled = disabled;

      if (s.status === "running") {
        const elapsed = ((Date.now() / 1000) - s.t_start).toFixed(1);
        statusArea.innerHTML = `
          <div class="status-running">
            <div class="status-text pulse">Teleop active...</div>
            <div class="status-detail">
              Episode ${s.episode_idx} &middot;
              "${escHtml(s.instruction)}" &middot;
              ${elapsed}s elapsed
            </div>
          </div>`;
      } else {
        statusArea.innerHTML = "";
      }

      if (s.history && s.history.length > 0) {
        if (s.history.length !== lastHistoryLen) {
          lastHistoryLen = s.history.length;
          historyList.innerHTML = s.history.map(h => {
            const lbl = h.labels || {};
            const parts = [];
            if (lbl.valid === true) parts.push("valid");
            if (lbl.valid === false) parts.push("invalid");
            if (lbl.success === true) parts.push("success");
            if (lbl.success === false) parts.push("fail");
            if (lbl.score != null) parts.push("score:" + lbl.score);
            if (lbl.notes) parts.push(lbl.notes);
            const lblStr = parts.length ? parts.join(" · ") : "";
            const expTag = h.experiment_name ? `<span class="history-tag exp">${escHtml(h.experiment_name)}</span>` : "";
            const taskTag = h.task_name ? `<span class="history-tag task">${escHtml(h.task_name)}</span>` : "";
            return `
            <div class="history-item" id="hi_${h.episode_idx}">
              <div class="history-row" onclick="toggleLabelForm(${h.episode_idx})">
                <span class="history-idx">#${h.episode_idx}</span>
                <span class="history-instruction">"${escHtml(h.instruction)}"</span>
                <span class="history-stats">${h.steps} rec steps, ${h.duration_s.toFixed(1)}s</span>
                ${expTag}${taskTag}
              </div>
              ${lblStr ? '<div class="history-labels">' + escHtml(lblStr) + '</div>' : ''}
              <div class="label-form" id="lf_${h.episode_idx}">
                <div class="label-row">
                  <span class="label-field"><label>Experiment</label>
                    <input class="label-input label-input-wide" type="text" id="lexp_${h.episode_idx}"
                      value="${escAttr(h.experiment_name || '')}" />
                  </span>
                  <span class="label-field"><label>Task</label>
                    <input class="label-input label-input-wide" type="text" id="ltask_${h.episode_idx}"
                      value="${escAttr(h.task_name || '')}" />
                  </span>
                </div>
                <div class="label-row">
                  <span class="label-field"><label>Valid</label>
                    <span class="label-toggle">
                      <button id="lv_y_${h.episode_idx}" onclick="setLabel(${h.episode_idx},'valid',true)">Y</button>
                      <button id="lv_n_${h.episode_idx}" onclick="setLabel(${h.episode_idx},'valid',false)">N</button>
                    </span>
                  </span>
                  <span class="label-field"><label>Success</label>
                    <span class="label-toggle">
                      <button id="ls_y_${h.episode_idx}" onclick="setLabel(${h.episode_idx},'success',true)">Y</button>
                      <button id="ls_n_${h.episode_idx}" onclick="setLabel(${h.episode_idx},'success',false)">N</button>
                    </span>
                  </span>
                  <span class="label-field"><label>Score</label>
                    <input class="label-input" type="number" step="any" id="lsc_${h.episode_idx}"
                      value="${lbl.score != null ? lbl.score : ''}" />
                  </span>
                </div>
                <div class="label-row">
                  <span class="label-field"><label>Notes</label>
                    <input class="label-input label-input-wide" type="text" id="ln_${h.episode_idx}"
                      value="${escAttr(lbl.notes || '')}" placeholder="optional" />
                  </span>
                  <button class="label-save" onclick="saveLabels(${h.episode_idx})">Save</button>
                  <span id="lmsg_${h.episode_idx}"></span>
                </div>
              </div>
            </div>`}).join("");

          for (const h of s.history) {
            const lbl = h.labels || {};
            if (lbl.valid === true) { const e = document.getElementById("lv_y_"+h.episode_idx); if(e) e.className += " active-yes"; }
            if (lbl.valid === false) { const e = document.getElementById("lv_n_"+h.episode_idx); if(e) e.className += " active-no"; }
            if (lbl.success === true) { const e = document.getElementById("ls_y_"+h.episode_idx); if(e) e.className += " active-yes"; }
            if (lbl.success === false) { const e = document.getElementById("ls_n_"+h.episode_idx); if(e) e.className += " active-no"; }
          }
        }
      } else {
        lastHistoryLen = 0;
        historyList.innerHTML = '<div class="history-empty">No episodes yet.</div>';
      }
    }

    function toggleLabelForm(idx) {
      const el = document.getElementById("lf_" + idx);
      if (el) el.classList.toggle("open");
    }
    window.toggleLabelForm = toggleLabelForm;

    function setLabel(idx, field, value) {
      if (!pendingLabels[idx]) pendingLabels[idx] = {};
      if (pendingLabels[idx][field] === value) {
        pendingLabels[idx][field] = null;
      } else {
        pendingLabels[idx][field] = value;
      }
      if (field === "valid") {
        const y = document.getElementById("lv_y_" + idx);
        const n = document.getElementById("lv_n_" + idx);
        if(y) y.className = pendingLabels[idx].valid === true ? "active-yes" : "";
        if(n) n.className = pendingLabels[idx].valid === false ? "active-no" : "";
      }
      if (field === "success") {
        const y = document.getElementById("ls_y_" + idx);
        const n = document.getElementById("ls_n_" + idx);
        if(y) y.className = pendingLabels[idx].success === true ? "active-yes" : "";
        if(n) n.className = pendingLabels[idx].success === false ? "active-no" : "";
      }
    }
    window.setLabel = setLabel;

    async function saveLabels(idx) {
      const lbl = pendingLabels[idx] || {};
      const scoreEl = document.getElementById("lsc_" + idx);
      const notesEl = document.getElementById("ln_" + idx);
      const expNameEl = document.getElementById("lexp_" + idx);
      const taskNameEl = document.getElementById("ltask_" + idx);
      const scoreVal = scoreEl && scoreEl.value.trim() !== "" ? parseFloat(scoreEl.value) : null;
      const notesVal = notesEl ? notesEl.value.trim() || null : null;
      const body = {
        episode_idx: idx,
        valid: lbl.valid != null ? lbl.valid : null,
        success: lbl.success != null ? lbl.success : null,
        score: scoreVal,
        notes: notesVal,
        experiment_name: expNameEl ? expNameEl.value.trim() : null,
        task_name: taskNameEl ? taskNameEl.value.trim() : null,
      };
      try {
        const r = await fetch("/episode/label", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(body),
        });
        const msg = document.getElementById("lmsg_" + idx);
        if (r.ok) {
          if (msg) { msg.className = "label-saved"; msg.textContent = "Saved"; setTimeout(() => msg.textContent = "", 2000); }
          lastHistoryLen = 0;
        } else {
          const err = await r.json();
          if (msg) { msg.className = ""; msg.style.color = "#f85149"; msg.textContent = err.detail || "Error"; }
        }
      } catch(e) { console.error(e); }
    }
    window.saveLabels = saveLabels;

    function escHtml(s) {
      const d = document.createElement("div");
      d.textContent = s || "";
      return d.innerHTML;
    }
    function escAttr(s) {
      return String(s || "").replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
    }

    async function startEpisode() {
      if (currentStatus === "running") return;
      const instruction = instructionEl.value.trim();
      const walltimeRaw = walltimeEl.value.trim();
      let actionStepLimit = -1;
      if (walltimeRaw !== "" && configRateHz) {
        const wt = parseFloat(walltimeRaw);
        if (Number.isFinite(wt) && wt > 0) {
          actionStepLimit = Math.ceil(wt * configRateHz);
        }
      }
      const body = {
        instruction,
        experiment: experimentEl.value.trim(),
        task: taskEl.value.trim(),
        action_step_limit: actionStepLimit,
      };
      try {
        const r = await fetch("/episode/start", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(body)
        });
        if (!r.ok) {
          const err = await r.json();
          alert("Failed to start: " + (err.detail || JSON.stringify(err)));
        }
      } catch(e) { alert("Error: " + e); }
    }

    async function stopEpisode() {
      if (currentStatus !== "running") return;
      try { await fetch("/episode/stop", {method: "POST"}); } catch(e) {}
    }

    instructionEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        startEpisode();
      }
    });

    async function selectPhase(phase) {
      if (currentStatus !== "running") return;
      const error = document.getElementById("phase_error");
      try {
        const r = await fetch("/episode/phase", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({phase_index: phase})
        });
        const result = await r.json();
        if (!r.ok) throw new Error(result.detail || "Phase update failed");
        document.getElementById("current_phase").textContent = result.phase_index;
        error.textContent = "";
      } catch(e) { error.textContent = e.message; }
    }
    for (let phase = 0; phase < 10; phase++) {
      const button = document.createElement("button");
      button.textContent = phase;
      button.disabled = true;
      button.onclick = () => selectPhase(phase);
      document.getElementById("phase_buttons").appendChild(button);
    }

    document.addEventListener("keydown", (e) => {
      if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
      if (/^[0-9]$/.test(e.key) && !e.target.matches("input, textarea, select, [contenteditable]")) {
        e.preventDefault();
        selectPhase(Number(e.key));
      }
      if (e.key === "Escape") {
        e.preventDefault();
        stopEpisode();
      }
    });
  </script>
</body>
</html>
"""


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="teleop_service")

_state: AppState | None = None


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=LANDING_PAGE_HTML)


@app.get("/config")
def get_config():
    s = _state
    if s is None:
        raise HTTPException(500, "not initialized")
    return {
        "rate_hz": s.session.rate_hz,
        "record_rate_hz": s.session.record_rate_hz,
        "min_z": s.session.min_z,
        "record": s.session.record,
        "dry_run": s.session.dry_run,
        "so101_port": s.so101_port,
        "gripper_initialized": s.gripper_initialized,
        "base_run_dir": s.base_run_dir,
    }


@app.post("/min_z")
def set_min_z(body: dict):
    """Update the safety z floor between episodes."""
    global _state
    s = _state
    if s is None:
        raise HTTPException(500, "not initialized")
    try:
        new_min_z = float(body.get("min_z"))
    except (TypeError, ValueError):
        raise HTTPException(400, "min_z must be a number")
    with s.lock:
        if s.status == "running":
            raise HTTPException(409, "cannot change min_z while running")
        # TeleopSessionConfig is frozen — rebuild with the new value.
        s.session = replace(s.session, min_z=new_min_z)
    return {"ok": True, "min_z": new_min_z}


@app.get("/status")
def get_status():
    s = _state
    if s is None:
        return {"status": "error", "detail": "not initialized"}
    with s.lock:
        return {
            "status": s.status,
            "phase_index": s.phases.get(),
            "episode_idx": s.episode_idx,
            "instruction": s.current_config.instruction if s.current_config else "",
            "t_start": s.t_start,
            "history": [
                {
                    "episode_idx": h.episode_idx,
                    "instruction": h.instruction,
                    "steps": h.steps,
                    "duration_s": h.duration_s,
                    "experiment_name": h.experiment_name,
                    "task_name": h.task_name,
                    "labels": h.labels,
                }
                for h in s.history[:20]
            ],
        }


@app.post("/episode/start")
def episode_start(body: dict):
    s = _state
    if s is None:
        raise HTTPException(500, "not initialized")

    instruction = (body.get("instruction") or "").strip()
    experiment = (body.get("experiment") or "").strip() or None
    task = (body.get("task") or "").strip()
    try:
        action_step_limit = int(body.get("action_step_limit", -1))
    except (TypeError, ValueError):
        action_step_limit = -1

    with s.lock:
        if s.status == "running":
            raise HTTPException(409, "episode already running")

        config = EpisodeConfig(
            instruction=instruction,
            task=task,
            action_step_limit=action_step_limit,
            experiment=experiment,
        )
        s.phases = PhaseSelector()
        s.current_config = config
        s.status = "running"
        s.t_start = time.time()
        s.stop_event.clear()
        s.current_result = None

        t = threading.Thread(target=_run_episode_thread, args=(s,), daemon=True)
        s.episode_thread = t
        t.start()

    return {"ok": True, "episode_idx": s.episode_idx}


@app.post("/episode/phase")
def episode_phase(body: dict):
    s = _state
    if s is None:
        raise HTTPException(500, "not initialized")
    with s.lock:
        if s.status != "running":
            raise HTTPException(409, "no episode running")
        try:
            s.phases.set(body.get("phase_index"))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "phase_index": s.phases.get()}


@app.post("/episode/stop")
def episode_stop():
    s = _state
    if s is None:
        raise HTTPException(500, "not initialized")
    s.stop_event.set()
    return {"ok": True}


@app.post("/episode/label")
def episode_label(body: dict):
    """Update labels + experiment/task names for a completed episode."""
    s = _state
    if s is None:
        raise HTTPException(500, "not initialized")

    episode_idx = body.get("episode_idx")
    if episode_idx is None:
        raise HTTPException(400, "episode_idx is required")

    with s.lock:
        hist = next((h for h in s.history if h.episode_idx == episode_idx), None)
    if hist is None:
        raise HTTPException(404, f"episode {episode_idx} not found in history")

    labels: dict[str, Any] = {}
    if body.get("valid") is not None:
        labels["valid"] = bool(body["valid"])
    if body.get("success") is not None:
        labels["success"] = bool(body["success"])
    if body.get("score") is not None:
        labels["score"] = float(body["score"])
    if body.get("notes") is not None:
        labels["notes"] = str(body["notes"])

    new_exp = body.get("experiment_name")
    new_task = body.get("task_name")

    with s.lock:
        hist.labels.update(labels)
        if new_exp is not None:
            hist.experiment_name = str(new_exp)
        if new_task is not None:
            hist.task_name = str(new_task)

    if hist.episode_dir:
        meta_path = os.path.join(hist.episode_dir, "meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["valid"] = hist.labels.get("valid")
                meta["success"] = hist.labels.get("success")
                meta["score"] = hist.labels.get("score")
                meta["episode_notes"] = hist.labels.get("notes")
                if new_exp is not None:
                    meta["experiment"] = hist.experiment_name or None
                if new_task is not None:
                    meta["task"] = hist.task_name or None
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                raise HTTPException(500, f"failed to update meta.json: {e}")

    return {
        "ok": True,
        "labels": hist.labels,
        "experiment_name": hist.experiment_name,
        "task_name": hist.task_name,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global _state

    p = argparse.ArgumentParser(description="Web UI for SO-101 teleop data collection")
    p.add_argument("--web-port", type=int, default=DEFAULT_PORT,
        help=f"Web server port (default: {DEFAULT_PORT})")
    p.add_argument("--port", default="/dev/ttyACM0", help="SO-101 leader serial port")
    p.add_argument("--franky-service-url", default=FRANKY_SERVICE_URL,
        help="Franky service URL (robot control + URDF fetch)")
    p.add_argument(
        "--gripper",
        choices=["none", "robotiq", "franka"],
        default="robotiq",
        help="Gripper backend: robotiq (local gripper_service), franka (Hand via franky_service), none",
    )
    p.add_argument("--no-gripper", action="store_true", help="Alias for --gripper none")
    p.add_argument("--gripper-service-url", default=GRIPPER_SERVICE_URL,
        help="Robotiq gripper_service URL (ignored when --gripper franka)")
    p.add_argument("--left-camera-id", default=LEFT_CAMERA_SERIAL,
        help="Left camera serial/id (default: LEFT_CAMERA_SERIAL)")
    p.add_argument("--wrist-camera-id", default=WRIST_CAMERA_SERIAL,
        help="Wrist camera serial/id (default: WRIST_CAMERA_SERIAL)")
    p.add_argument("--right-camera-id", default=RIGHT_CAMERA_SERIAL,
        help="Right camera serial/id (default: RIGHT_CAMERA_SERIAL)")
    p.add_argument("--rate-hz", type=float, default=100.0, help="Control loop rate (Hz)")
    p.add_argument("--min-z", type=float, default=DEFAULT_MIN_EE_Z,
        help=f"Min EE Z height (m) — table safety (default: {DEFAULT_MIN_EE_Z})")
    p.add_argument("--record", default=True, action="store_true", help="Record episodes")
    p.add_argument("--record-rate-hz", type=float, default=15.0, help="Recording rate (Hz)")
    p.add_argument("--record-jpeg-quality", type=int, default=RECORD_JPEG_QUALITY,
        help=f"JPEG quality for recorded images (default: {RECORD_JPEG_QUALITY})")
    p.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality for camera snapshots")
    p.add_argument("--output-dir", default="output", help="Base output directory for recordings")
    p.add_argument("--task", default="", help="Task name (used for the run directory)")
    p.add_argument("--dry-run", action="store_true",
        help="Read SO-101 but do not command the robot or gripper")
    args = p.parse_args()

    gripper_backend = "none" if args.no_gripper else str(args.gripper)
    if gripper_backend == "franka":
        gripper_url = str(args.franky_service_url)
    else:
        gripper_url = str(args.gripper_service_url)

    session = TeleopSessionConfig(
        rate_hz=float(args.rate_hz),
        record_rate_hz=float(args.record_rate_hz),
        jpeg_quality=int(args.jpeg_quality),
        record_jpeg_quality=int(args.record_jpeg_quality),
        min_z=float(args.min_z),
        dry_run=bool(args.dry_run),
        record=bool(args.record),
    )

    print("Initializing robot...")
    print(f"Franky service: {args.franky_service_url}")
    print(f"Gripper backend: {gripper_backend}"
          + (f" ({gripper_url})" if gripper_backend != "none" else ""))
    droid = DroidPlus(
        franky_service_url=args.franky_service_url,
        gripper_service_url=gripper_url,
        left_camera_serial=args.left_camera_id,
        wrist_camera_serial=args.wrist_camera_id,
        right_camera_serial=args.right_camera_id,
    )
    wait_for_cameras(droid)

    try:
        pin_model, pin_data, ee_frame = build_fk_model(args.franky_service_url)
    except Exception as e:
        print(f"WARNING: FK unavailable ({e}) — Z safety DISABLED")
        pin_model = pin_data = ee_frame = None

    try:
        droid.stop()
    except Exception:
        pass

    gripper_initialized = False
    if gripper_backend != "none" and not args.dry_run:
        gripper_initialized = init_gripper(droid, backend=gripper_backend)
    else:
        reason = "dry-run" if args.dry_run else "--gripper none"
        print(f"Skipping gripper initialization ({reason}).")

    settle_s = 2.0 if (gripper_initialized and gripper_backend == "robotiq") else 0.0
    teleop = connect_so101(args.port, settle_s=settle_s)

    base_run_dir: str | None = None
    if session.record:
        base_run_dir = make_teleop_run_dir(args.task, parent=args.output_dir)

    _state = AppState(
        droid=droid,
        teleop=teleop,
        gripper_initialized=gripper_initialized,
        pin_model=pin_model,
        pin_data=pin_data,
        ee_frame=ee_frame,
        session=session,
        so101_port=args.port,
        base_run_dir=base_run_dir,
    )

    print("\n  teleop_service")
    print(f"    UI:            http://localhost:{args.web_port}")
    print(f"    SO-101:        {args.port}")
    print(f"    Gripper:       {'initialized' if gripper_initialized else 'disabled'}")
    print(f"    Control rate:  {session.rate_hz} Hz")
    print(f"    Record rate:   {session.record_rate_hz} Hz")
    print(f"    Record:        {base_run_dir or 'disabled'}")
    print(f"    Dry run:       {session.dry_run}")
    print("\n  Terminal: ESC to stop episode, Ctrl+C to quit.\n")

    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs=dict(app=app, host="0.0.0.0", port=args.web_port, log_level="warning"),
        daemon=True,
    )
    server_thread.start()

    try:
        with KeyPoller() as keys:
            while True:
                ch = keys.poll_char()
                if ch == "\x1b" and _state.status == "running":
                    print("\n[terminal] ESC → stopping episode...")
                    _state.stop_event.set()
                if ch is not None and ch in "0123456789":
                    with _state.lock:
                        if _state.status == "running":
                            _state.phases.set(int(ch))
                            print(f"[phase] {ch}", flush=True)
                if ch is None:
                    time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nShutting down...")
        if _state and _state.status == "running":
            _state.stop_event.set()
            if _state.episode_thread:
                _state.episode_thread.join(timeout=5)

    if gripper_initialized:
        try:
            droid.gripper.shutdown_async()
        except Exception:
            pass
        if gripper_backend == "franka":
            try:
                droid.gripper.disconnect()
                print("Franka Hand connection released.")
            except Exception as e:
                print(f"Warning: could not release Franka Hand: {e}")


if __name__ == "__main__":
    main()
