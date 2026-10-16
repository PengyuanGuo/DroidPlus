# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference-client runtime for droid-plus.

Mirrors ``robolab_policy_client/runtime.py``: a ``POLICY_REGISTRY`` maps
backend names to client classes, and :func:`create_client` looks the backend
up and constructs it with signature-filtered kwargs.

Droid-plus currently ships a single Pi0 client; the registry is small but
follows the same shape so adding a real-robot variant of another backend is
mechanical.
"""

import inspect
from typing import Any

from droid_plus.eval.base_client import InferenceClient

# Populate the registry with whatever backends can be imported. A missing
# backend dep (e.g. openpi_client not installed) excludes that backend without
# breaking the import of the runtime itself.
POLICY_REGISTRY: dict[str, type[InferenceClient]] = {}

try:
    from .pi0_droid import Pi0DroidJointposClient
    for _name in ("pi0", "pi0_fast", "pi05", "paligemma", "paligemma_fast"):
        POLICY_REGISTRY[_name] = Pi0DroidJointposClient
except ImportError:
    pass

try:
    from .dreamzero import DreamZeroClient
    POLICY_REGISTRY["dreamzero"] = DreamZeroClient
except ImportError:
    pass

try:
    from .diffusion_droid import DiffusionDroidClient
    POLICY_REGISTRY["diffusion"] = DiffusionDroidClient
except ImportError:
    pass


def create_client(name: str, **kwargs: Any) -> InferenceClient:
    """Construct the inference client for a given backend name.

    Kwargs whose values are ``None`` are dropped so the client's own defaults
    apply. Kwargs not declared by the client's ``__init__`` are silently
    ignored.

    Raises:
        ValueError: If ``name`` is not a registered backend.
    """
    name = name.lower()
    try:
        cls = POLICY_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unsupported policy '{name}'. Known: {sorted(POLICY_REGISTRY)}"
        ) from None

    candidate = {"policy_variant": name, **kwargs}
    accepted = set(inspect.signature(cls.__init__).parameters)
    filtered = {k: v for k, v in candidate.items() if k in accepted and v is not None}
    return cls(**filtered)
