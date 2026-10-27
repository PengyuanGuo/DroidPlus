#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Web UI experiment runner — browser interface for policy episodes with recording.

Usage:
    python scripts/run_experiment_ui.py
    python scripts/run_experiment_ui.py --dry-run --port 54324
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from droid_plus.analysis.end_effector_pose import compute_and_save_ee_trajectory_single
from droid_plus.constants import DEFAULT_POLICY, POLICIES, RECORD_JPEG_QUALITY, FRANKY_SERVICE_URL, GRIPPER_SERVICE_URL
from droid_plus.eval.episode_runner import (
    EpisodeConfig,
    EpisodeResult,
    SessionConfig,
    finalize_episode_recording,
    run_episode,
)
from droid_plus.eval.experiment_setup import (
    build_policy_client,
    init_droid_and_home,
    make_base_run_dir,
    maybe_compute_ee_trajectories,
    wait_for_cameras,
)
from droid_plus.logging import EpisodeRecorder
from droid_plus.robot import DroidPlus
from droid_plus.utils import KeyPoller

DEFAULT_PORT = 54324


# ── Shared state ─────────────────────────────────────────────────────────────

@dataclass
class EpisodeHistory:
    episode_idx: int
    instruction: str
    policy: str
    steps: int
    duration_s: float
    episode_dir: str | None = None
    experiment_name: str = ""
    task_name: str = ""
    labels: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppState:
    droid: DroidPlus
    session: SessionConfig
    open_loop_horizon: int
    base_run_dir: str | None
    diffusion_policy_path: str | None = None
    # Lazy-init per canonical policy name; evicted on override.
    policy_clients: dict[str, Any] = field(default_factory=dict)
    # Per-policy host/port/display-name overrides for POLICIES entries (session-scoped).
    policy_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    # User-added policies via "Add new" (session-scoped). Keyed by URI.
    custom_policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Episode control
    episode_idx: int = 0
    episode_thread: threading.Thread | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    current_result: EpisodeResult | None = None
    current_config: EpisodeConfig | None = None
    current_policy: str = ""
    status: str = "idle"  # "idle" | "running"
    t_start: float = 0.0
    history: list[EpisodeHistory] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _effective_host_port(state: AppState, policy_name: str) -> tuple[str, int]:
    """Return host/port for a policy, checking overrides, POLICIES, then custom_policies."""
    if policy_name == "diffusion" and state.diffusion_policy_path:
        return "local", 0
    host: Any = None
    port: Any = None
    if policy_name in POLICIES:
        ov = state.policy_overrides.get(policy_name, {})
        cfg = POLICIES[policy_name]
        host = ov.get("host") or cfg.get("host")
        port = ov.get("port") or cfg.get("port")
    elif policy_name in state.custom_policies:
        cfg = state.custom_policies[policy_name]
        host = cfg.get("host")
        port = cfg.get("port")
    if not host or port is None:
        raise ValueError(f"No host/port configured for policy '{policy_name}'")
    return str(host), int(port)


def _display_name(state: AppState, policy_name: str) -> str:
    if policy_name == "diffusion" and state.diffusion_policy_path:
        return f"diffusion ({Path(state.diffusion_policy_path).name})"
    if policy_name in POLICIES:
        return state.policy_overrides.get(policy_name, {}).get("display_name") or policy_name
    if policy_name in state.custom_policies:
        return state.custom_policies[policy_name].get("display_name") or policy_name
    return policy_name


def _policy_exists(state: AppState, policy_name: str) -> bool:
    if policy_name == "diffusion" and state.diffusion_policy_path:
        return True
    return policy_name in POLICIES or policy_name in state.custom_policies


def _get_or_create_policy(state: AppState, policy_name: str) -> Any:
    """Get or lazily create a policy client for the given policy name."""
    if policy_name in state.policy_clients:
        return state.policy_clients[policy_name]
    if policy_name == "diffusion" and state.diffusion_policy_path:
        from droid_plus.policies.diffusion_droid import DiffusionDroidClient
        client = DiffusionDroidClient(state.diffusion_policy_path)
        state.policy_clients[policy_name] = client
        return client
    host, port = _effective_host_port(state, policy_name)
    client = build_policy_client(host, port, state.open_loop_horizon)
    state.policy_clients[policy_name] = client
    return client


def _run_episode_thread(state: AppState) -> None:
    """Background thread that runs a single episode."""
    try:
        recorder: EpisodeRecorder | None = None
        if state.session.record and state.base_run_dir is not None:
            recorder = EpisodeRecorder(
                base_run_dir=state.base_run_dir,
                episode_idx=state.episode_idx,
                jpeg_quality=state.session.record_jpeg_quality,
            )

        policy = _get_or_create_policy(state, state.current_policy)
        host, port = _effective_host_port(state, state.current_policy)
        episode_session = replace(
            state.session,
            policy_host=host,
            policy_port=port,
            policy_name=state.current_policy,
        )

        result = run_episode(
            config=state.current_config,
            session=episode_session,
            droid=state.droid,
            policy=policy,
            episode_idx=state.episode_idx,
            recorder=recorder,
            should_stop=state.stop_event.is_set,
        )

        with state.lock:
            state.current_result = result

        if not state.session.dry_run:
            state.droid.move_to_home()
            try:
                state.droid.gripper.open_async(wait=False)
            except Exception:
                pass

        if state.session.record and result.recorder is not None:
            finalize_episode_recording(
                result=result,
                config=state.current_config,
                session=episode_session,
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
                policy=state.current_policy,
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
  <title>droid+ service</title>
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

    .policy-options { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }
    .policy-btn { background: #0b111a; color: #e6edf3; border: 2px solid #223042; border-radius: 10px; padding: 10px 18px; font-size: 15px; font-weight: 600; cursor: pointer; transition: border-color 0.15s, background 0.15s; }
    .policy-btn:hover { border-color: #1f6feb; background: #111d2e; }
    .policy-btn.selected { border-color: #1f6feb; background: #122040; color: #58a6ff; }
    .policy-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .policy-btn-add { border-style: dashed; color: #9fb2c8; }
    .policy-btn-add.selected { border-color: #3fb950; background: #0d2818; color: #3fb950; border-style: solid; }

    .policy-edit { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; font-size: 13px; margin-top: 8px; padding-top: 10px; border-top: 1px dashed #223042; }
    .pol-field { display: flex; flex-direction: column; flex: 1 1 180px; min-width: 0; }
    .pol-field-port { flex: 0 1 110px; }
    .pol-field label { color: #9fb2c8; margin-bottom: 4px; font-size: 12px; }
    .pol-field input { background: #0b111a; color: #e6edf3; border: 1px solid #223042; border-radius: 6px; padding: 6px 10px; font-size: 13px; font-family: inherit; }
    .pol-field input:focus { outline: none; border-color: #1f6feb; }
    .policy-edit-actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    .policy-save { background: #1f6feb; color: #fff; border: none; border-radius: 6px; padding: 6px 14px; font-size: 13px; font-weight: 600; cursor: pointer; }
    .policy-save:hover { background: #388bfd; }
    .policy-saved { color: #3fb950; font-size: 12px; }
    .policy-err { color: #f85149; font-size: 12px; }

    .main { flex: 1; display: flex; flex-direction: column; align-items: center; padding: 24px; gap: 16px; }

    .field-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 20px; }
    .field-cell label { display: block; font-size: 12px; color: #9fb2c8; margin-bottom: 4px; }
    .field-cell .hint { font-size: 11px; color: #6b7f97; font-weight: 400; margin-left: 6px; }
    .field-cell .hint code { font-size: 10px; }
    .field-input { width: 100%; box-sizing: border-box; background: #0b111a; color: #e6edf3; border: 1px solid #223042; border-radius: 6px; padding: 6px 10px; font-size: 13px; font-family: inherit; outline: none; transition: border-color 0.15s; }
    .field-input:focus { border-color: #1f6feb; }
    .field-input:disabled { opacity: 0.5; cursor: not-allowed; }

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
    <div class="header-title">droid+ service</div>
    <span id="status_badge" class="badge badge-idle">idle</span>
    <div class="muted" style="margin-left: auto;">Episode <code id="episode_idx">0</code></div>
  </div>

  <div class="main">
    <div class="card">
      <div class="card-label">Policy</div>
      <div class="policy-options" id="policy_options"></div>
      <div class="policy-edit" id="policy_edit">
        <div class="pol-field" id="pol_uri_row" style="display: none;">
          <label>URI</label>
          <input type="text" id="pol_uri" placeholder="e.g. pi0_custom" />
        </div>
        <div class="pol-field">
          <label>Display name</label>
          <input type="text" id="pol_name" />
        </div>
        <div class="pol-field">
          <label>Host</label>
          <input type="text" id="pol_host" placeholder="policy-server.local" />
        </div>
        <div class="pol-field pol-field-port">
          <label>Port</label>
          <input type="number" id="pol_port" placeholder="8000" />
        </div>
        <div class="policy-edit-actions">
          <button class="policy-save" id="pol_save_btn" onclick="savePolicy()">Save</button>
          <span id="pol_msg"></span>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-label">Episode</div>
      <div class="instruction-row">
        <textarea id="instruction" class="instruction-input" rows="3"
          placeholder="Type an instruction for the robot..." autofocus></textarea>
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
          <input type="text" id="task_name" class="field-input" placeholder="e.g. BananaInBowl" />
        </div>
        <div class="field-cell">
          <label>Max action steps <span class="hint">Blank = unlimited.</span></label>
          <input type="number" id="action_step_limit" class="field-input" placeholder="-1" />
        </div>
        <div class="field-cell">
          <label>Max walltime (s) <span class="hint">at <span id="rate_display">?</span> Hz. (Update via CLI with <code>--rate-hz</code>)</span></label>
          <input type="number" id="max_walltime" class="field-input" placeholder="unlimited" step="0.1" />
        </div>
      </div>
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
    const stepLimitEl = document.getElementById("action_step_limit");
    const walltimeEl = document.getElementById("max_walltime");
    const rateDisplayEl = document.getElementById("rate_display");
    const policyContainer = document.getElementById("policy_options");
    const polNameEl = document.getElementById("pol_name");
    const polHostEl = document.getElementById("pol_host");
    const polPortEl = document.getElementById("pol_port");
    const polUriEl = document.getElementById("pol_uri");
    const polUriRow = document.getElementById("pol_uri_row");
    const polSaveBtn = document.getElementById("pol_save_btn");
    const polMsgEl = document.getElementById("pol_msg");
    const badgeEl = document.getElementById("status_badge");
    const episodeIdxEl = document.getElementById("episode_idx");
    const statusArea = document.getElementById("status_area");
    const historyList = document.getElementById("history_list");

    let currentStatus = "idle";
    let selectedPolicy = "";
    let policyInfo = {};   // {name: {host, port, display_name, is_custom}}
    let policyEditMode = "edit";  // "edit" | "add"
    let configRateHz = null;     // launch-time rate, fetched from /config
    let lastHistoryLen = 0;
    const pendingLabels = {};  // episode_idx -> {valid, success, score, notes, experiment_name, task_name}

    // ── Restore persisted fields ──
    const LS_KEY = "droidplus_service_state_v1";
    function loadPersisted() {
      try {
        const raw = localStorage.getItem(LS_KEY);
        if (!raw) return;
        const s = JSON.parse(raw);
        if (s.experiment) experimentEl.value = s.experiment;
        if (s.task) taskEl.value = s.task;
        if (s.step_limit != null) stepLimitEl.value = s.step_limit;
      } catch(e) {}
    }
    function savePersisted() {
      try {
        localStorage.setItem(LS_KEY, JSON.stringify({
          experiment: experimentEl.value,
          task: taskEl.value,
          step_limit: stepLimitEl.value,
        }));
      } catch(e) {}
    }
    [experimentEl, taskEl, stepLimitEl].forEach(el => {
      el.addEventListener("input", savePersisted);
    });
    loadPersisted();

    // ── Step ↔ walltime sync (driven by launch-time rate_hz) ──
    let syncingLimits = false;
    function stepsFromWalltime(wt) {
      if (!configRateHz || !Number.isFinite(wt) || wt <= 0) return null;
      return Math.ceil(wt * configRateHz);
    }
    function walltimeFromSteps(st) {
      if (!configRateHz || !Number.isFinite(st) || st <= 0) return null;
      return +(st / configRateHz).toFixed(2);
    }
    stepLimitEl.addEventListener("input", () => {
      if (syncingLimits) return;
      syncingLimits = true;
      const raw = stepLimitEl.value.trim();
      if (raw === "" || raw === "-1") {
        walltimeEl.value = "";
      } else {
        const steps = parseInt(raw, 10);
        const wt = walltimeFromSteps(steps);
        walltimeEl.value = wt != null ? wt : "";
      }
      syncingLimits = false;
      savePersisted();
    });
    walltimeEl.addEventListener("input", () => {
      if (syncingLimits) return;
      syncingLimits = true;
      const raw = walltimeEl.value.trim();
      if (raw === "") {
        stepLimitEl.value = "";
      } else {
        const wt = parseFloat(raw);
        const steps = stepsFromWalltime(wt);
        stepLimitEl.value = steps != null ? steps : "";
      }
      syncingLimits = false;
      savePersisted();
    });

    async function loadConfig() {
      try {
        const r = await fetch("/config");
        const d = await r.json();
        configRateHz = d.rate_hz;
        if (rateDisplayEl) rateDisplayEl.textContent = configRateHz;
        // Derive walltime from restored step_limit now that rate is known.
        if (stepLimitEl.value && !walltimeEl.value) {
          const steps = parseInt(stepLimitEl.value, 10);
          const wt = walltimeFromSteps(steps);
          if (wt != null) walltimeEl.value = wt;
        }
      } catch(e) { console.error("Failed to load config", e); }
    }
    loadConfig();

    // ── Init: load policies ──
    async function loadPolicies() {
      try {
        const r = await fetch("/policies");
        const d = await r.json();
        policyInfo = {};
        for (const p of d.policies) policyInfo[p.name] = p;
        if (!selectedPolicy || !policyInfo[selectedPolicy]) selectedPolicy = d.default;
        policyContainer.innerHTML = "";
        for (const p of d.policies) {
          const btn = document.createElement("button");
          btn.className = "policy-btn" + (policyEditMode === "edit" && p.name === selectedPolicy ? " selected" : "");
          btn.textContent = p.display_name || p.name;
          btn.dataset.policy = p.name;
          btn.onclick = () => selectPolicy(p.name);
          policyContainer.appendChild(btn);
        }
        const addBtn = document.createElement("button");
        addBtn.className = "policy-btn policy-btn-add" + (policyEditMode === "add" ? " selected" : "");
        addBtn.textContent = "+ Add new";
        addBtn.dataset.policy = "__add__";
        addBtn.onclick = () => startAddPolicy();
        policyContainer.appendChild(addBtn);
        if (policyEditMode === "edit") refreshPolicyEditor();
      } catch(e) { console.error("Failed to load policies", e); }
    }
    loadPolicies();

    function refreshPolicyEditor() {
      const p = policyInfo[selectedPolicy] || {};
      polUriRow.style.display = "none";
      polUriEl.value = "";
      polNameEl.value = p.display_name || selectedPolicy;
      polHostEl.value = p.host || "";
      polPortEl.value = p.port || "";
      polSaveBtn.textContent = "Save";
      polMsgEl.textContent = "";
      polMsgEl.className = "";
    }

    function selectPolicy(name) {
      if (currentStatus === "running") return;
      policyEditMode = "edit";
      selectedPolicy = name;
      for (const btn of policyContainer.children) {
        btn.classList.toggle("selected", btn.dataset.policy === name);
      }
      refreshPolicyEditor();
    }

    function startAddPolicy() {
      if (currentStatus === "running") return;
      policyEditMode = "add";
      for (const btn of policyContainer.children) {
        btn.classList.toggle("selected", btn.dataset.policy === "__add__");
      }
      polUriRow.style.display = "";
      polUriEl.value = "";
      polNameEl.value = "";
      polHostEl.value = "";
      polPortEl.value = "";
      polSaveBtn.textContent = "Add policy";
      polMsgEl.textContent = "";
      polMsgEl.className = "";
      polUriEl.focus();
    }
    window.startAddPolicy = startAddPolicy;

    async function savePolicy() {
      if (currentStatus === "running") return;
      if (policyEditMode === "add") return addPolicy();
      return saveOverride();
    }
    window.savePolicy = savePolicy;

    async function saveOverride() {
      const body = {
        name: selectedPolicy,
        host: polHostEl.value.trim(),
        port: polPortEl.value ? parseInt(polPortEl.value, 10) : null,
        display_name: polNameEl.value.trim(),
      };
      try {
        const r = await fetch("/policies/override", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(body),
        });
        if (r.ok) {
          polMsgEl.className = "policy-saved";
          polMsgEl.textContent = "Saved";
          setTimeout(() => { polMsgEl.textContent = ""; }, 2000);
          await loadPolicies();
        } else {
          const err = await r.json();
          polMsgEl.className = "policy-err";
          polMsgEl.textContent = err.detail || "Error";
        }
      } catch(e) {
        polMsgEl.className = "policy-err";
        polMsgEl.textContent = "Network error";
      }
    }

    async function addPolicy() {
      const body = {
        uri: polUriEl.value.trim(),
        host: polHostEl.value.trim(),
        port: polPortEl.value ? parseInt(polPortEl.value, 10) : null,
        display_name: polNameEl.value.trim(),
      };
      if (!body.uri || !body.host || !body.port) {
        polMsgEl.className = "policy-err";
        polMsgEl.textContent = "URI, host, and port are required";
        return;
      }
      try {
        const r = await fetch("/policies/add", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(body),
        });
        if (r.ok) {
          const d = await r.json();
          selectedPolicy = d.name;
          policyEditMode = "edit";
          polMsgEl.className = "policy-saved";
          polMsgEl.textContent = "Added";
          setTimeout(() => { polMsgEl.textContent = ""; }, 2000);
          await loadPolicies();
        } else {
          const err = await r.json();
          polMsgEl.className = "policy-err";
          polMsgEl.textContent = err.detail || "Error";
        }
      } catch(e) {
        polMsgEl.className = "policy-err";
        polMsgEl.textContent = "Network error";
      }
    }

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
      stepLimitEl.disabled = disabled;
      walltimeEl.disabled = disabled;
      polNameEl.disabled = disabled;
      polHostEl.disabled = disabled;
      polPortEl.disabled = disabled;
      polUriEl.disabled = disabled;
      polSaveBtn.disabled = disabled;
      for (const btn of policyContainer.children) btn.disabled = disabled;

      if (s.status === "running") {
        const elapsed = ((Date.now() / 1000) - s.t_start).toFixed(1);
        statusArea.innerHTML = `
          <div class="status-running">
            <div class="status-text pulse">Executing...</div>
            <div class="status-detail">
              Episode ${s.episode_idx} &middot;
              "${escHtml(s.instruction)}" &middot;
              ${elapsed}s elapsed &middot;
              policy: ${escHtml(s.policy)}
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
                <span class="history-stats">${h.steps} steps, ${h.duration_s.toFixed(1)}s</span>
                ${expTag}${taskTag}
                <span class="history-tag">${escHtml(h.policy)}</span>
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
      const instruction = instructionEl.value.trim();
      if (!instruction) return;
      if (currentStatus === "running") return;
      const stepLimitRaw = stepLimitEl.value.trim();
      const stepLimit = stepLimitRaw === "" ? -1 : parseInt(stepLimitRaw, 10);
      const body = {
        instruction,
        policy: selectedPolicy,
        experiment: experimentEl.value.trim(),
        task: taskEl.value.trim(),
        action_step_limit: Number.isFinite(stepLimit) ? stepLimit : -1,
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

    document.addEventListener("keydown", (e) => {
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

app = FastAPI(title="droidplus_service")

_state: AppState | None = None


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=LANDING_PAGE_HTML)


@app.get("/config")
def get_config():
    """Launch-time configuration the UI needs (currently just rate_hz)."""
    s = _state
    if s is None:
        raise HTTPException(500, "not initialized")
    return {
        "rate_hz": s.session.rate_hz,
        "dry_run": s.session.dry_run,
    }


@app.get("/policies")
def get_policies():
    s = _state
    if s is None:
        raise HTTPException(500, "not initialized")
    policies = []
    for name in POLICIES.keys():
        host, port = _effective_host_port(s, name)
        policies.append({
            "name": name,
            "host": host,
            "port": port,
            "display_name": _display_name(s, name),
            "is_custom": False,
        })
    for name in s.custom_policies.keys():
        host, port = _effective_host_port(s, name)
        policies.append({
            "name": name,
            "host": host,
            "port": port,
            "display_name": _display_name(s, name),
            "is_custom": True,
        })
    if s.diffusion_policy_path and not any(p["name"] == "diffusion" for p in policies):
        policies.append({
            "name": "diffusion", "host": "local", "port": 0,
            "display_name": _display_name(s, "diffusion"), "is_custom": False,
        })
    return {"policies": policies, "default": s.session.policy_name}


@app.post("/policies/add")
def policies_add(body: dict):
    """Add a session-scoped custom policy by URI."""
    s = _state
    if s is None:
        raise HTTPException(500, "not initialized")
    uri = str(body.get("uri") or "").strip()
    host = str(body.get("host") or "").strip()
    port = body.get("port")
    display_name = str(body.get("display_name") or "").strip() or None

    if not uri:
        raise HTTPException(400, "uri is required")
    if not host:
        raise HTTPException(400, "host is required")
    if port is None or port == "":
        raise HTTPException(400, "port is required")
    try:
        port_int = int(port)
    except (TypeError, ValueError):
        raise HTTPException(400, "port must be an integer")

    with s.lock:
        if s.status == "running":
            raise HTTPException(409, "cannot add policy while running")
        if _policy_exists(s, uri):
            raise HTTPException(409, f"policy '{uri}' already exists")
        s.custom_policies[uri] = {
            "host": host,
            "port": port_int,
            "display_name": display_name,
        }
    return {"ok": True, "name": uri}


@app.post("/policies/override")
def policies_override(body: dict):
    s = _state
    if s is None:
        raise HTTPException(500, "not initialized")
    name = body.get("name")
    if name == "diffusion" and s.diffusion_policy_path:
        raise HTTPException(400, "Local diffusion checkpoint: restart with --policy-path to change it")
    if not name or name not in POLICIES:
        raise HTTPException(400, f"unknown policy: {name}")
    with s.lock:
        if s.status == "running":
            raise HTTPException(409, "cannot edit policy while running")
        ov = s.policy_overrides.setdefault(name, {})
        host = body.get("host")
        port = body.get("port")
        display_name = body.get("display_name")
        if host is not None:
            ov["host"] = str(host).strip() or None
        if port is not None:
            ov["port"] = int(port) if port != "" else None
        if display_name is not None:
            ov["display_name"] = str(display_name).strip() or None
        # Drop Nones to keep overrides tidy.
        for k in list(ov.keys()):
            if ov[k] is None:
                del ov[k]
        # Evict cached client so next episode rebuilds against the new endpoint.
        s.policy_clients.pop(name, None)
    return {"ok": True, "override": s.policy_overrides.get(name, {})}


@app.get("/status")
def get_status():
    s = _state
    if s is None:
        return {"status": "error", "detail": "not initialized"}
    with s.lock:
        result = {
            "status": s.status,
            "episode_idx": s.episode_idx,
            "instruction": s.current_config.instruction if s.current_config else "",
            "policy": s.current_policy,
            "t_start": s.t_start,
            "history": [
                {
                    "episode_idx": h.episode_idx,
                    "instruction": h.instruction,
                    "policy": h.policy,
                    "steps": h.steps,
                    "duration_s": h.duration_s,
                    "experiment_name": h.experiment_name,
                    "task_name": h.task_name,
                    "labels": h.labels,
                }
                for h in s.history[:20]
            ],
        }
    return result


@app.post("/episode/start")
def episode_start(body: dict):
    s = _state
    if s is None:
        raise HTTPException(500, "not initialized")

    instruction = (body.get("instruction") or "").strip()
    policy_name = body.get("policy") or s.session.policy_name
    experiment = (body.get("experiment") or "").strip() or None
    task = (body.get("task") or "").strip()
    try:
        action_step_limit = int(body.get("action_step_limit", -1))
    except (TypeError, ValueError):
        action_step_limit = -1

    if not instruction:
        raise HTTPException(400, "instruction is required")
    if not _policy_exists(s, policy_name):
        raise HTTPException(400, f"unknown policy: {policy_name}")

    with s.lock:
        if s.status == "running":
            raise HTTPException(409, "episode already running")

        config = EpisodeConfig(
            instruction=instruction,
            task=task,
            action_step_limit=action_step_limit,
            experiment=experiment,
        )
        s.current_config = config
        s.current_policy = policy_name
        s.status = "running"
        s.t_start = time.time()
        s.stop_event.clear()
        s.current_result = None

        t = threading.Thread(target=_run_episode_thread, args=(s,), daemon=True)
        s.episode_thread = t
        t.start()

    return {"ok": True, "episode_idx": s.episode_idx}


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

    p = argparse.ArgumentParser(description="Web UI for running policy experiments")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Web server port (default: {DEFAULT_PORT})")
    p.add_argument("--franky-service-url", default=FRANKY_SERVICE_URL,
        help="Franky service URL for arm control and the Franka Hand")
    p.add_argument("--gripper", choices=("franka", "robotiq"), default="franka",
        help="Gripper backend (default: franka, through the Franky service)")
    p.add_argument("--gripper-service-url", default=GRIPPER_SERVICE_URL,
        help="Robotiq service URL; ignored when --gripper franka")
    p.add_argument("--policy", default=DEFAULT_POLICY, help=f"Default policy (available: {', '.join(POLICIES.keys())})")
    p.add_argument("--policy-path", type=Path,
        help="Local LeRobot diffusion checkpoint directory; required for --policy diffusion")
    p.add_argument("--open-loop-horizon", type=int, default=10, help="Policy chunk horizon")
    p.add_argument("--rate-hz", type=float, default=15.0, help="Control loop rate (Hz)")
    p.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality for camera snapshots")
    p.add_argument("--record-jpeg-quality", type=int, default=RECORD_JPEG_QUALITY,
        help=f"JPEG quality for recorded images (default: {RECORD_JPEG_QUALITY}, from constants.py)")
    p.add_argument("--dry-run", action="store_true", help="Run policy but do not command the robot")
    p.add_argument("--record", default=True, action="store_true", help="Record episodes")
    p.add_argument("--output-dir", default="output",
        help="Base output directory for recordings")
    args = p.parse_args()
    if args.policy == "diffusion" and args.policy_path is None:
        p.error("--policy diffusion requires --policy-path pointing to a checkpoint directory")
    if args.policy_path is not None:
        args.policy_path = args.policy_path.expanduser().resolve()
        config_path = args.policy_path / "config.json"
        if not config_path.is_file() or not (args.policy_path / "model.safetensors").is_file():
            p.error("--policy-path must contain config.json and model.safetensors (choose dp_050000, not dp_ckpts)")
        if json.loads(config_path.read_text()).get("type") != "diffusion":
            p.error("--policy-path must be a LeRobot diffusion checkpoint")

    record_jpeg_quality = args.record_jpeg_quality

    session = SessionConfig(
        policy_host="",  # resolved per-episode from overrides/POLICIES
        policy_port=0,
        rate_hz=float(args.rate_hz),
        jpeg_quality=int(args.jpeg_quality),
        record_jpeg_quality=int(record_jpeg_quality),
        dry_run=bool(args.dry_run),
        record=bool(args.record),
        policy_name=str(args.policy),
    )

    print("Initializing robot...")
    gripper_url = args.franky_service_url if args.gripper == "franka" else args.gripper_service_url
    print(f"Gripper backend: {args.gripper} ({gripper_url})")
    droid = DroidPlus(franky_service_url=args.franky_service_url, gripper_service_url=gripper_url)
    wait_for_cameras(droid)

    base_run_dir: str | None = None
    if session.record:
        base_run_dir = make_base_run_dir("ui", args.policy, parent=args.output_dir)

    init_droid_and_home(droid, dry_run=session.dry_run)

    _state = AppState(
        droid=droid,
        session=session,
        open_loop_horizon=int(args.open_loop_horizon),
        base_run_dir=base_run_dir,
        diffusion_policy_path=str(args.policy_path) if args.policy_path else None,
        current_policy=args.policy,
    )

    print("\n  droidplus_service")
    print(f"    UI:      http://localhost:{args.port}")
    print(f"    Policy:  {args.policy} (changeable in UI)")
    print(f"    Record:  {base_run_dir or 'disabled'}")
    print(f"    Dry run: {args.dry_run}")
    print("\n  Terminal: ESC to stop episode, Ctrl+C to quit.\n")

    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs=dict(app=app, host="0.0.0.0", port=args.port, log_level="warning"),
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
                if ch is None:
                    time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nShutting down...")
        if _state and _state.status == "running":
            _state.stop_event.set()
            if _state.episode_thread:
                _state.episode_thread.join(timeout=5)

    if not session.dry_run:
        try:
            droid.gripper.shutdown_async()
        except Exception:
            pass

    if session.record and _state.episode_idx > 0:
        maybe_compute_ee_trajectories(base_run_dir)


if __name__ == "__main__":
    main()
