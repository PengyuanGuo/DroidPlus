"""
Phase-filtered dataset for PandaTabletop with point conditioning.

Loads an annotated zarr (from panda_tabletop_phase_annotator) and
provides samples only from the requested phases. Each contiguous
phase segment becomes a virtual episode so the SequenceSampler
never crosses phase boundaries.

Point-conditioned phases supply ``point`` (2-D normalised pixel);
non-point phases supply NaN (caller should not use the point).
"""

import copy
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import zarr
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

from atomic_policy.dataset.panda_tabletop_image_dataset import (
    _load_or_build_resized_dual_cache,
)


class _PhaseSegmentedBuffer:
    """Thin wrapper that overrides episode_ends of a ReplayBuffer.

    Splits original episodes at phase-label transitions so each
    contiguous phase segment is treated as its own episode.
    """

    def __init__(self, replay_buffer: ReplayBuffer, phase_labels: np.ndarray):
        self._rb = replay_buffer
        original_ends = replay_buffer.episode_ends[:]
        original_starts = np.concatenate([[0], original_ends[:-1]])

        new_ends = []
        for es, ee in zip(original_starts, original_ends):
            es, ee = int(es), int(ee)
            labels = phase_labels[es:ee]
            for i in range(1, len(labels)):
                if labels[i] != labels[i - 1]:
                    new_ends.append(es + i)
            new_ends.append(ee)

        self.episode_ends = np.array(sorted(set(new_ends)), dtype=np.int64)
        self._phase_labels = phase_labels

    @property
    def n_episodes(self):
        return len(self.episode_ends)

    def keys(self):
        return self._rb.keys()

    def __getitem__(self, key):
        return self._rb[key]

    def get_episode_phase(self, ep_idx: int) -> int:
        """Return the phase label for a given virtual episode."""
        start = 0 if ep_idx == 0 else int(self.episode_ends[ep_idx - 1])
        return int(self._phase_labels[start])


class PandaTabletopPhaseDataset(BaseImageDataset):
    """Phase-filtered dataset with optional point conditioning.

    Parameters
    ----------
    zarr_path : str
        Path to annotated zarr (must contain phase_label, phase_prompt, point_pixel).
    target_phases : list of int
        Which phases to include, e.g. [0,2,4] for grasp, [3] for place, [1,5] for drawer.
    has_point : bool
        Whether this dataset provides point conditioning.
    """

    def __init__(
        self,
        zarr_path: str,
        target_phases: List[int],
        has_point: bool = False,
        horizon: int = 16,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        image_shape: Optional[Tuple[int, int]] = None,
        image_cache_zarr_path: Optional[str] = None,
    ):
        super().__init__()

        self.target_phases = set(target_phases)
        self.has_point = has_point

        # Load non-image data into replay buffer
        load_keys = ['state', 'action', 'phase_label', 'phase_prompt']
        if has_point:
            load_keys.append('point_pixel')
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=load_keys)

        # Load phase labels (already in replay buffer)
        phase_labels = self.replay_buffer['phase_label'][:]

        # Build segmented buffer
        self._seg_buffer = _PhaseSegmentedBuffer(self.replay_buffer, phase_labels)

        # Build episode mask: include only segments whose phase is in target_phases
        n_virtual_eps = self._seg_buffer.n_episodes
        episode_mask = np.zeros(n_virtual_eps, dtype=bool)
        for ep_idx in range(n_virtual_eps):
            if self._seg_buffer.get_episode_phase(ep_idx) in self.target_phases:
                episode_mask[ep_idx] = True

        # Apply val split *only on episodes that match this slot's phases* so
        # small target_phases sets still get a representative validation slice.
        eligible_idx = np.flatnonzero(episode_mask)
        val_mask_local = get_val_mask(n_episodes=len(eligible_idx), val_ratio=val_ratio, seed=seed)
        val_mask = np.zeros(n_virtual_eps, dtype=bool)
        val_mask[eligible_idx[val_mask_local]] = True
        train_mask = ~val_mask & episode_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        # Lazy image loading. Optionally source images (and cache) from a
        # different zarr that contains byte-identical agent_view/wrist_cam
        # (e.g. the raw pre-annotation dataset) to avoid rebuilding the
        # resized cache for each annotated variant.
        image_zarr_path = image_cache_zarr_path if image_cache_zarr_path is not None else zarr_path
        zarr_root = zarr.open(zarr_path, 'r')
        image_zarr_root = zarr.open(image_zarr_path, 'r') if image_cache_zarr_path is not None else zarr_root
        self._zarr_agent_view_raw = image_zarr_root['data']['agent_view']
        self._zarr_wrist_cam_raw = image_zarr_root['data']['wrist_cam']
        assert self._zarr_agent_view_raw.shape[0] == zarr_root['data']['action'].shape[0], (
            f'image_cache_zarr_path images ({self._zarr_agent_view_raw.shape[0]}) '
            f'must have same length as zarr_path action ({zarr_root["data"]["action"].shape[0]})'
        )

        # Image cache (same as PandaTabletopImageDataset)
        self._use_cache = False
        if image_shape is not None:
            src_h = self._zarr_agent_view_raw.shape[1]
            src_w = self._zarr_agent_view_raw.shape[2]
            if (image_shape[0] != src_h) or (image_shape[1] != src_w):
                av_cache, wc_cache = _load_or_build_resized_dual_cache(
                    image_zarr_path,
                    self._zarr_agent_view_raw,
                    self._zarr_wrist_cam_raw,
                    image_shape[0],
                    image_shape[1],
                )
                self._zarr_agent_view = av_cache
                self._zarr_wrist_cam = wc_cache
                self._use_cache = True

        if not self._use_cache:
            self._zarr_agent_view = self._zarr_agent_view_raw
            self._zarr_wrist_cam = self._zarr_wrist_cam_raw

        # Build sampler on segmented buffer
        self.sampler = SequenceSampler(
            replay_buffer=self._seg_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        # True for virtual episodes belonging to this slot's target phases AND
        # picked as validation by get_val_mask. (was previously inverted, which
        # silently produced empty val sets.)
        self.val_episode_mask = val_mask & episode_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.image_shape = image_shape

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_mask = self.val_episode_mask
        val_set.sampler = SequenceSampler(
            replay_buffer=self._seg_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
        )
        val_set.train_mask = val_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        kwargs.pop('dataset', None)
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],
        }
        if self.has_point:
            # Only fit normalizer on valid points (sentinel -1 = no point)
            points = self.replay_buffer['point_pixel'][:]
            valid = points[:, 0] >= 0
            if np.any(valid):
                data['point'] = points[valid]
            else:
                data['point'] = np.zeros((1, 2), dtype=np.float32)

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        normalizer['wrist_image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _load_zarr_sequence(self, zarr_arr, idx):
        """Load a sequence from zarr using the sampler's index logic."""
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.sampler.indices[idx]
        sample = zarr_arr[buffer_start_idx:buffer_end_idx]
        if (sample_start_idx > 0) or (sample_end_idx < self.horizon):
            data = np.zeros(
                shape=(self.horizon,) + zarr_arr.shape[1:],
                dtype=zarr_arr.dtype,
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < self.horizon:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
            return data
        return sample

    def _sample_to_data(self, sample):
        if self._use_cache:
            agent_view = sample['agent_view'].astype(np.float32) / 255.0
            wrist_cam = sample['wrist_cam'].astype(np.float32) / 255.0
        else:
            agent_view = np.moveaxis(sample['agent_view'], -1, 1).astype(np.float32) / 255.0
            wrist_cam = np.moveaxis(sample['wrist_cam'], -1, 1).astype(np.float32) / 255.0
            if self.image_shape is not None:
                agent_view = self._resize_image(agent_view)
                wrist_cam = self._resize_image(wrist_cam)

        agent_pos = sample['state'].astype(np.float32)

        # Phase prompt from first timestep
        prompt_bytes = sample['phase_prompt'][0]
        prompt_str = bytes(prompt_bytes).rstrip(b'\x00').decode('utf-8')

        data = {
            'obs': {
                'image': agent_view,
                'wrist_image': wrist_cam,
                'agent_pos': agent_pos,
            },
            'action': sample['action'].astype(np.float32),
            'prompt': prompt_str,
        }

        if self.has_point:
            # point_pixel from first timestep (same for all steps in a phase)
            point = sample['point_pixel'][0].astype(np.float32)
            # Replace sentinel (-1,-1) with zeros for non-grounded phases
            if point[0] < 0:
                point = np.zeros(2, dtype=np.float32)
            data['point'] = point

        return data

    def _resize_image(self, image):
        if self.image_shape is None:
            return image
        target_h, target_w = self.image_shape
        T, C, H, W = image.shape
        if (H, W) == (target_h, target_w):
            return image
        resized = np.empty((T, C, target_h, target_w), dtype=image.dtype)
        for t in range(T):
            img_hwc = np.moveaxis(image[t], 0, -1)
            img_r = cv2.resize(img_hwc, (target_w, target_h), interpolation=cv2.INTER_AREA)
            resized[t] = np.moveaxis(img_r, -1, 0)
        return resized

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        # Lazy-load images
        sample['agent_view'] = self._load_zarr_sequence(self._zarr_agent_view, idx)
        sample['wrist_cam'] = self._load_zarr_sequence(self._zarr_wrist_cam, idx)
        data = self._sample_to_data(sample)

        prompt_str = data.pop('prompt')
        point = data.pop('point', None)

        torch_data = dict_apply(data, torch.from_numpy)
        torch_data['prompt'] = prompt_str
        if point is not None:
            torch_data['point'] = torch.from_numpy(point)

        return torch_data
