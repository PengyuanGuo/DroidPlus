# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LeRobot diffusion policy client for DroidPlus teleop checkpoints.

The checkpoint was trained on 8-D absolute joint targets (7 joints + gripper)
and three cameras: left, right, wrist. ``episode_runner`` sends the returned
action as a Franka joint-position target.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from droid_plus.eval.base_client import InferenceClient

__all__ = ["DiffusionDroidClient"]


class DiffusionDroidClient(InferenceClient):
    """Local diffusion policy. One ``infer_chunk`` call returns one action.

    LeRobot's ``select_action`` keeps the observation history (``n_obs_steps``)
    and the action queue (``n_action_steps``). Calling it once per control tick
    matches that rollout.
    """

    def __init__(self, pretrained_path: str) -> None:
        super().__init__()
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = PreTrainedConfig.from_pretrained(pretrained_path)
        cfg.device = device
        print(f"[DiffusionDroidClient] Loading {pretrained_path} on {device}...")
        self.policy = DiffusionPolicy.from_pretrained(pretrained_path, config=cfg)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": device},
            },
        )
        self._warned_black_right = False
        print("[DiffusionDroidClient] Checkpoint ready.")

    def reset(self, *, env_id: int | None = None) -> None:
        super().reset(env_id=env_id)
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def infer_chunk(self, obs: dict, instruction: str, *, env_id: int = 0) -> np.ndarray:
        del env_id
        action = self._query_server(self._pack_request(self._extract_observation(obs), instruction))
        return np.asarray(action, dtype=np.float32).reshape(1, -1)

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        del env_id
        left = np.asarray(raw_obs["left_image"], dtype=np.uint8)
        wrist = np.asarray(raw_obs["wrist_image"], dtype=np.uint8)
        right = raw_obs.get("right_image")
        if right is None:
            right = np.zeros_like(left)
            if not self._warned_black_right:
                print("[DiffusionDroidClient] No right camera in the observation; sending a black right image.")
                self._warned_black_right = True
        else:
            right = np.asarray(right, dtype=np.uint8)
        joints = np.asarray(raw_obs["joint_position"], dtype=np.float32).reshape(-1)
        gripper = float(raw_obs.get("gripper_position", 0.0))
        state = np.concatenate([joints, np.array([gripper], dtype=np.float32)])
        return {
            "observation.images.left": left,
            "observation.images.right": right,
            "observation.images.wrist": wrist,
            "observation.state": state,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict[str, Any]:
        return {"observation": extracted_obs, "instruction": instruction}

    def _query_server(self, request: dict) -> np.ndarray:
        import torch
        from lerobot.policies.utils import prepare_observation_for_inference

        device = torch.device(self.policy.config.device)
        batch = prepare_observation_for_inference(
            dict(request["observation"]),
            device,
            task=request.get("instruction") or "",
        )
        batch = self.preprocessor(batch)
        with torch.inference_mode():
            action = self.policy.select_action(batch)
            action = self.postprocessor(action)
        if isinstance(action, torch.Tensor):
            values = action.detach().float().cpu().numpy()
        else:
            values = np.asarray(action)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.shape[0] < 8:
            raise RuntimeError(f"Diffusion action must have 8 dims, got {values.shape}")
        return values[:8]

    def _unpack_response(self, response: np.ndarray) -> np.ndarray:
        return np.asarray(response, dtype=np.float32).reshape(1, -1)
