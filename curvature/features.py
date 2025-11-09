# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn

import numpy as np

try:
    from utils_hyp import PoincarePlaneDistance  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PoincarePlaneDistance = None


@dataclass
class FeatureBatch:
    """Container for latent representations captured from the shared trunk."""

    tensor: torch.Tensor

    def as_numpy(self) -> np.ndarray:
        return self.tensor.detach().cpu().numpy()


class SharedRepresentationExtractor:
    """
    Runs the IMPALA-style shared trunk up to (but not including) the actor/critic heads.

    This class works for both fully shared networks (single trunk feeding both heads)
    and architectures with explicit actor/critic heads (in which case we simply return
    the output of the shared trunk).
    """

    def __init__(self, actor_critic: nn.Module):
        if not hasattr(actor_critic, "sm"):
            raise ValueError("Actor-critic model is missing shared modules (`sm`).")
        self.actor_critic = actor_critic
        self.device = next(actor_critic.parameters()).device
        self._head_start_idx = self._locate_head_boundary()

    def _head_types(self) -> Sequence[type]:
        head_types = [nn.Linear]
        if PoincarePlaneDistance is not None:
            head_types.append(PoincarePlaneDistance)
        return tuple(head_types)

    def _locate_head_boundary(self) -> int:
        """
        Determine the index in actor_critic.sm at which the head begins.

        For fully shared architectures, we stop before the final linear/Poincare layer.
        For partially shared architectures, the shared trunk already ends before heads,
        so we keep all modules.
        """
        modules = list(self.actor_critic.sm.children())
        if not getattr(self.actor_critic, "fully_shared", False):
            return len(modules)
        head_types = self._head_types()
        for idx in range(len(modules) - 1, -1, -1):
            if isinstance(modules[idx], head_types):
                return idx
        # Fallback: keep the whole sequence if no head module was found.
        return len(modules)

    def __call__(self, obs: torch.Tensor) -> FeatureBatch:
        """
        Compute the shared representation for a batch of observations.

        Args:
            obs: Tensor with shape [batch, channels, height, width] already on
                 the correct device (Preprocessor takes care of this).
        """
        if obs.device != self.device:
            obs = obs.to(self.device)
        with torch.no_grad():
            out = obs
            for idx, module in enumerate(self.actor_critic.sm.children()):
                if idx >= self._head_start_idx:
                    break
                out = module(out)
            return FeatureBatch(out)

    def feature_dim(self, sample_shape: Optional[Sequence[int]] = None) -> int:
        """
        Utility helper to determine flattened feature dimension.

        If sample_shape is provided, a dummy tensor is used. Otherwise, the
        dimension is inferred from the most recent call.
        """
        if sample_shape is None:
            raise ValueError("sample_shape is required to infer feature_dim.")
        dummy = torch.zeros((1, *sample_shape), device=self.device)
        feat = self(dummy)
        return feat.tensor.view(1, -1).shape[-1]
