from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from curvature.features import SharedRepresentationExtractor


@dataclass
class CoverageHyperParams:
    """
    Hyperparameters implementing Section 2 of curvature/PLAN.md.
    """

    n_seeds: int = 500
    steps_per_seed: int = 10_000
    subsample_stride: int = 4
    epsilon: float = 0.20
    sticky_prob: float = 0.15
    # Procgen specifics
    num_levels: int = 1
    render_mode: Optional[str] = None


@dataclass
class CoverageResult:
    """
    Stored coverage data ready for clustering.
    """

    features: np.ndarray  # [N, D] latent activations (float32)
    transitions: np.ndarray  # [M, 2] indices into features array
    seeds: np.ndarray  # [N] seed id for each recorded state
    timesteps: np.ndarray  # [N] environment step index for each recorded state
    stride: int


class CoverageCollector:
    """
    Executes the coverage rollout policy and records latent representations.
    """

    def __init__(
        self,
        env_cfg: DictConfig,
        agent,
        preprocessor,
        feature_extractor: SharedRepresentationExtractor,
        n_actions: int,
        hyperparams: CoverageHyperParams,
        rng_seed: int = 0,
    ):
        self._env_cfg = env_cfg
        self._agent = agent
        self._preprocessor = preprocessor
        self._feature_extractor = feature_extractor
        self._n_actions = n_actions
        self._hp = hyperparams
        self._rng = np.random.default_rng(rng_seed)
        self._agent.ac_model.eval()

    def _instantiate_env(self, seed: int):
        """
        Instantiate a ProcgenGym3Env that always starts from the provided seed.
        """
        env = hydra.utils.instantiate(
            self._env_cfg,
            num=1,
            start_level=seed,
            num_levels=self._hp.num_levels,
            render_mode=self._hp.render_mode,
        )
        return env

    def _choose_action(
        self, obs_tensor: torch.Tensor, prev_action: Optional[int]
    ) -> Tuple[int, Optional[int]]:
        """
        Apply epsilon-random + sticky action noise on top of the policy.
        """
        if prev_action is not None and self._rng.random() < self._hp.sticky_prob:
            return prev_action, prev_action

        if self._rng.random() < self._hp.epsilon:
            sampled = int(self._rng.integers(low=0, high=self._n_actions))
            return sampled, sampled

        with torch.no_grad():
            act = self._agent.act(obs_tensor, det=False)
        sampled = int(np.asarray(act).reshape(-1)[0])
        return sampled, sampled

    def _tensor_from_obs(self, obs_rgb: np.ndarray) -> torch.Tensor:
        channel_first = obs_rgb.transpose(0, 3, 1, 2).astype(np.float32)
        return self._preprocessor.preprocess_obs(channel_first)

    def collect(self) -> CoverageResult:
        """
        Run coverage rollouts for n_seeds and return latent states + transitions.
        """
        feature_list: List[np.ndarray] = []
        transitions: List[Tuple[int, int]] = []
        seeds: List[int] = []
        timesteps: List[int] = []

        total_states = 0
        start_time = time.time()
        sampled_seeds = self._rng.integers(low=0, high=1_000_000, size=self._hp.n_seeds)
        for seed_value in sampled_seeds:
            env = self._instantiate_env(int(seed_value))
            try:
                rew, obs, first = env.observe()
                prev_action: Optional[int] = None
                last_recorded_idx: Optional[int] = None
                for step in range(self._hp.steps_per_seed):
                    obs_tensor = self._tensor_from_obs(obs["rgb"])
                    action, prev_action = self._choose_action(obs_tensor, prev_action)
                    if step % self._hp.subsample_stride == 0:
                        feature = (
                            self._feature_extractor(obs_tensor)
                            .tensor.reshape(obs_tensor.shape[0], -1)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        feature_list.append(feature[0])
                        seeds.append(int(seed_value))
                        timesteps.append(step)
                        current_idx = len(feature_list) - 1
                        if last_recorded_idx is not None:
                            transitions.append((last_recorded_idx, current_idx))
                        last_recorded_idx = current_idx
                        total_states += 1

                    env.act(np.array([action], dtype=np.int32))
                    rew, obs, first = env.observe()
                    if first[0]:
                        prev_action = None
                        last_recorded_idx = None
            finally:
                env.close()

        features = np.stack(feature_list, axis=0).astype(np.float32)
        transitions_arr = (
            np.array(transitions, dtype=np.int64) if transitions else np.zeros((0, 2), dtype=np.int64)
        )
        result = CoverageResult(
            features=features,
            transitions=transitions_arr,
            seeds=np.array(seeds, dtype=np.int32),
            timesteps=np.array(timesteps, dtype=np.int32),
            stride=self._hp.subsample_stride,
        )
        elapsed = time.time() - start_time
        print(
            f"[coverage] collected {result.features.shape[0]} states "
            f"and {result.transitions.shape[0]} transitions in {elapsed:.1f}s."
        )
        return result
