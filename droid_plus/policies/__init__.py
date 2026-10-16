# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete policy inference clients for droid-plus plus factory / registry.

Each client module is a backend-specific subclass of
:class:`droid_plus.eval.base_client.InferenceClient`. :mod:`.runtime` owns
the ``POLICY_REGISTRY`` and ``create_client`` factory.

The Pi0 client requires ``openpi_client`` at runtime; its import is guarded
so this package stays importable even when that dep is missing.
"""

from .runtime import POLICY_REGISTRY, create_client

__all__ = ["POLICY_REGISTRY", "create_client"]

try:
    from .pi0_droid import Pi0DroidJointposClient as Pi0DroidJointposClient

    __all__.append("Pi0DroidJointposClient")
except ImportError:
    pass

try:
    from .dreamzero import DreamZeroClient as DreamZeroClient

    __all__.append("DreamZeroClient")
except ImportError:
    pass

try:
    from .diffusion_droid import DiffusionDroidClient as DiffusionDroidClient

    __all__.append("DiffusionDroidClient")
except ImportError:
    pass
