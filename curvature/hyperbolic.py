from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from curvature.pairs import PairDataset


@dataclass
class HyperbolicConfig:
    embedding_dim: int = 16
    c_min: float = 1e-3
    max_outer_steps: int = 50
    inner_steps: int = 10
    curvature_steps: int = 2
    batch_size: int = 4096
    embedding_lr: float = 1e-2
    curvature_lr: float = 2e-3
    tau: float = 1e-3
    rho: float = 0.9
    boundary_penalty: float = 1e-3
    min_relative_improvement: float = 0.0025  # 0.25%
    patience: int = 5
    curvature_lr_decay: float = 0.5
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    curvature_warmstart: float = 1.0


@dataclass
class HyperbolicFitResult:
    optimal_c: float
    alpha: float
    history: List[Dict[str, float]]
    best_state: Dict[str, np.ndarray]


class HyperbolicCurvatureEstimator:
    """
    Jointly optimizes node embeddings and curvature following curvature/PLAN.md.
    """

    def __init__(self, num_nodes: int, config: HyperbolicConfig):
        self.num_nodes = num_nodes
        self.config = config
        self.device = config.device
        self.embeddings = nn.Parameter(torch.zeros(num_nodes, config.embedding_dim, device=self.device))
        torch.nn.init.normal_(self.embeddings, mean=0.0, std=1e-3)
        theta_init = self._theta_from_curvature(config.curvature_warmstart)
        self.theta = nn.Parameter(torch.tensor(theta_init, device=self.device, dtype=torch.float32))
        self.alpha = 1.0

    def _theta_from_curvature(self, target_c: float) -> float:
        val = max(target_c - self.config.c_min, 1e-4)
        return math.log(math.exp(val) - 1.0)

    def current_c(self) -> torch.Tensor:
        return F.softplus(self.theta) + self.config.c_min

    def project_embeddings(self) -> None:
        c_value = float(self.current_c().detach().cpu())
        radius = (1.0 - self.config.tau) / math.sqrt(max(c_value, 1e-6))
        with torch.no_grad():
            norms = torch.norm(self.embeddings, dim=-1, keepdim=True)
            mask = norms > radius
            if mask.any():
                self.embeddings[mask] = self.embeddings[mask] / norms[mask] * radius

    def poincare_distance(self, idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
        c = self.current_c()
        xa = self.embeddings[idx_a]
        xb = self.embeddings[idx_b]
        diff = torch.sum((xa - xb) ** 2, dim=-1)
        na = torch.sum(xa ** 2, dim=-1)
        nb = torch.sum(xb ** 2, dim=-1)
        denom = (1 - c * na) * (1 - c * nb)
        denom = torch.clamp(denom, min=1e-6)
        argument = 1 + 2 * c * diff / denom
        argument = torch.clamp(argument, min=1 + 1e-6)
        return torch.arccosh(argument) / torch.sqrt(c)

    def _boundary_penalty(self, c: torch.Tensor) -> torch.Tensor:
        norms = torch.sum(self.embeddings ** 2, dim=-1)
        excess = torch.relu(c * norms - self.config.rho)
        return self.config.boundary_penalty * torch.mean(excess ** 2)

    def _prepare_datasets(self, pair_data: PairDataset):
        finite_mask = np.isfinite(pair_data.distances)
        pairs = torch.from_numpy(pair_data.pairs[finite_mask]).long().to(self.device)
        dists = torch.from_numpy(pair_data.distances[finite_mask]).float().to(self.device)
        weights = torch.from_numpy(pair_data.weights[finite_mask]).float().to(self.device)
        n = pairs.shape[0]
        if n == 0:
            raise ValueError("No finite target distances available for optimization.")
        split = int(0.9 * n)
        permutation = torch.randperm(n)
        train_idx = permutation[:split]
        val_idx = permutation[split:]
        dataset_train = TensorDataset(pairs[train_idx], dists[train_idx], weights[train_idx])
        dataset_val = TensorDataset(pairs[val_idx], dists[val_idx], weights[val_idx])
        loader_train = DataLoader(dataset_train, batch_size=self.config.batch_size, shuffle=True, drop_last=False)
        loader_full = DataLoader(dataset_train, batch_size=self.config.batch_size, shuffle=False, drop_last=False)
        loader_val = DataLoader(dataset_val, batch_size=self.config.batch_size, shuffle=False, drop_last=False)
        return loader_train, loader_full, loader_val

    @torch.no_grad()
    def _recompute_alpha(self, data_loader: DataLoader) -> float:
        numerator = torch.zeros(1, device=self.device)
        denominator = torch.zeros(1, device=self.device)
        for batch_pairs, target, _ in data_loader:
            d_model = self.poincare_distance(batch_pairs[:, 0], batch_pairs[:, 1])
            numerator += torch.sum(d_model * target)
            denominator += torch.sum(target ** 2)
        alpha = (numerator / (denominator + 1e-8)).item()
        self.alpha = max(alpha, 1e-6)
        return self.alpha

    def _embedding_step(self, batch_pairs, target, weight):
        d_model = self.poincare_distance(batch_pairs[:, 0], batch_pairs[:, 1])
        diff = d_model - self.alpha * target
        loss = torch.mean((diff ** 2) * weight)
        penalty = self._boundary_penalty(self.current_c())
        return loss + penalty

    def _relative_distortion(self, data_loader: DataLoader) -> float:
        numerator = torch.zeros(1, device=self.device)
        denominator = torch.zeros(1, device=self.device)
        with torch.no_grad():
            for batch_pairs, target, _ in data_loader:
                d_model = self.poincare_distance(batch_pairs[:, 0], batch_pairs[:, 1])
                numerator += torch.sum(torch.abs(d_model - self.alpha * target))
                denominator += torch.sum(target)
        return (numerator / (denominator + 1e-8)).item()

    def _curvature_loss(self, data_loader: DataLoader) -> torch.Tensor:
        losses = []
        for batch_pairs, target, _ in data_loader:
            d_model = self.poincare_distance(batch_pairs[:, 0], batch_pairs[:, 1])
            diff = d_model - self.alpha * target
            losses.append(torch.mean(diff ** 2))
        return torch.stack(losses).mean()

    def fit(self, pair_data: PairDataset) -> HyperbolicFitResult:
        train_loader, alpha_loader, val_loader = self._prepare_datasets(pair_data)
        self.project_embeddings()
        history: List[Dict[str, float]] = []
        embed_optimizer = torch.optim.Adam([self.embeddings], lr=self.config.embedding_lr)
        curvature_optimizer = torch.optim.Adam([self.theta], lr=self.config.curvature_lr)

        best_state = {"embeddings": self.embeddings.detach().cpu().numpy(), "theta": self.theta.detach().cpu().numpy()}
        best_rel = float("inf")
        best_c = float(self.current_c().item())
        best_alpha = self.alpha
        no_improve = 0
        consecutive_increase = 0
        prev_rel = None

        for outer in range(self.config.max_outer_steps):
            start = time.time()
            alpha = self._recompute_alpha(alpha_loader)
            last_loss = 0.0
            for _ in range(self.config.inner_steps):
                for batch_pairs, target, weight in train_loader:
                    embed_optimizer.zero_grad(set_to_none=True)
                    loss = self._embedding_step(batch_pairs, target, weight)
                    loss.backward()
                    embed_optimizer.step()
                    self.project_embeddings()
                    last_loss = float(loss.item())

            for _ in range(self.config.curvature_steps):
                curvature_optimizer.zero_grad(set_to_none=True)
                curv_loss = self._curvature_loss(val_loader)
                curv_loss.backward()
                curvature_optimizer.step()
                self.project_embeddings()

            rel = self._relative_distortion(val_loader)
            elapsed = time.time() - start
            history.append(
                {
                    "outer_step": outer,
                    "alpha": alpha,
                    "curvature": float(self.current_c().item()),
                    "rel_dist": rel,
                    "train_loss": last_loss,
                    "time": elapsed,
                }
            )

            if rel + self.config.min_relative_improvement < best_rel:
                best_rel = rel
                best_c = float(self.current_c().item())
                best_alpha = alpha
                best_state = {
                    "embeddings": self.embeddings.detach().cpu().numpy(),
                    "theta": self.theta.detach().cpu().numpy(),
                }
                no_improve = 0
            else:
                no_improve += 1

            if prev_rel is not None and rel > prev_rel:
                consecutive_increase += 1
            else:
                consecutive_increase = 0
            prev_rel = rel

            if consecutive_increase >= 3:
                for pg in curvature_optimizer.param_groups:
                    pg["lr"] *= self.config.curvature_lr_decay
                consecutive_increase = 0

            if no_improve >= self.config.patience:
                rel_change = (best_rel - rel) / max(best_rel, 1e-6)
                if abs(rel_change) < self.config.min_relative_improvement:
                    print("[hyperbolic] early stop triggered.")
                    break

        return HyperbolicFitResult(
            optimal_c=best_c,
            alpha=best_alpha,
            history=history,
            best_state=best_state,
        )
