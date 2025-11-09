#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from curvature.clustering import run_kmeans, whiten_features
from curvature.coverage import CoverageCollector, CoverageHyperParams
from curvature.features import SharedRepresentationExtractor
from curvature.graph import TransitionGraphBuilder
from curvature.hyperbolic import HyperbolicConfig, HyperbolicCurvatureEstimator
from curvature.pairs import PairSampler
from main_hydra import make_models


REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "cfgs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate best-fit hyperbolic curvature for a Procgen task.")
    parser.add_argument("--env", required=True, help="Procgen environment name, e.g., bigfish")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained PPO/HPPO checkpoint (.pt)")
    parser.add_argument("--agent", default="onpolicy/ppo_sn", help="Hydra agent config override (default: onpolicy/ppo_sn)")
    parser.add_argument("--output-dir", default="curvature_runs", help="Directory to store intermediate artifacts")
    parser.add_argument("--device", default=None, help="Optional torch device override (cpu / cuda)")
    parser.add_argument("--disable-cuda", action="store_true", help="Force Hydra config to run on CPU")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for reproducibility")

    # Coverage parameters
    parser.add_argument("--coverage-seeds", type=int, default=500, help="Number of Procgen seeds for coverage rollouts")
    parser.add_argument("--coverage-steps", type=int, default=10_000, help="Steps per seed (before subsampling)")
    parser.add_argument("--subsample-stride", type=int, default=4, help="Record every Nth frame")
    parser.add_argument("--epsilon", type=float, default=0.20, help="Random action probability during coverage")
    parser.add_argument("--sticky-prob", type=float, default=0.15, help="Sticky action probability during coverage")

    # Graph + clustering
    parser.add_argument("--n-clusters", type=int, default=10_000, help="Number of latent clusters / graph nodes")
    parser.add_argument("--kmeans-batch", type=int, default=8192, help="MiniBatchKMeans batch size")
    parser.add_argument("--num-pairs", type=int, default=1_000_000, help="Number of metric pairs for hyperbolic fitting")

    # Hyperbolic optimization
    parser.add_argument("--embedding-dim", type=int, default=16, help="Poincaré embedding dimensionality")
    parser.add_argument("--max-outer-steps", type=int, default=50, help="Outer curvature steps")
    parser.add_argument("--inner-steps", type=int, default=10, help="Embedding optimizer steps per outer iteration")
    parser.add_argument("--batch-size", type=int, default=4096, help="Pair batch size for training")
    parser.add_argument("--embedding-lr", type=float, default=1e-2, help="Embedding optimizer LR")
    parser.add_argument("--curvature-lr", type=float, default=2e-3, help="Curvature optimizer LR")

    parser.add_argument("--save-intermediates", action="store_true", help="Persist coverage, clustering, and pair data")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(args.output_dir) / f"{args.env}_{timestamp}")
    print(f"[setup] writing artifacts to {run_dir}")

    hydra_overrides = [
        f"env@_global_=gen/{args.env}",
        f"agent@_global_={args.agent}",
        "n_envs=1",
        "n_eval_envs=1",
    ]
    if args.disable_cuda:
        hydra_overrides.append("disable_cuda=true")

    with initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        cfg = compose(config_name="config", overrides=hydra_overrides)
        agent, _, env, _, preprocessor = make_models(cfg)

    try:
        loaded_preproc = agent.load(str(checkpoint_path))
        if loaded_preproc is not None:
            preprocessor = loaded_preproc
    finally:
        if hasattr(env, "close"):
            env.close()

    device_str = args.device or cfg.device or ("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu")
    device = torch.device(device_str)
    agent.ac_model.to(device=device)
    agent.ac_model.eval()
    feature_extractor = SharedRepresentationExtractor(agent.ac_model)

    coverage_hp = CoverageHyperParams(
        n_seeds=args.coverage_seeds,
        steps_per_seed=args.coverage_steps,
        subsample_stride=args.subsample_stride,
        epsilon=args.epsilon,
        sticky_prob=args.sticky_prob,
    )
    collector = CoverageCollector(
        env_cfg=cfg.env,
        agent=agent,
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        n_actions=int(cfg.n_actions),
        hyperparams=coverage_hp,
        rng_seed=args.seed,
    )
    coverage = collector.collect()
    if args.save_intermediates:
        np.savez_compressed(
            run_dir / "coverage.npz",
            features=coverage.features,
            transitions=coverage.transitions,
            seeds=coverage.seeds,
            timesteps=coverage.timesteps,
        )

    whitened, whitening_stats = whiten_features(coverage.features)
    clustering = run_kmeans(
        whitened,
        n_clusters=args.n_clusters,
        batch_size=args.kmeans_batch,
        seed=args.seed,
    )
    assignments = clustering.assignments
    if args.save_intermediates:
        np.savez_compressed(
            run_dir / "clustering.npz",
            assignments=assignments,
            centers=clustering.centers,
            mean=whitening_stats.mean,
            std=whitening_stats.std,
        )

    graph_builder = TransitionGraphBuilder(
        assignments=assignments,
        transitions=coverage.transitions,
        num_nodes=args.n_clusters,
    )
    graph = graph_builder.build()

    sampler = PairSampler(graph, rng_seed=args.seed)
    pair_data = sampler.sample_pairs(args.num_pairs)
    if pair_data.pairs.shape[0] == 0:
        raise RuntimeError("Failed to sample any metric pairs; graph may be disconnected.")
    if args.save_intermediates:
        np.savez_compressed(
            run_dir / "pairs.npz",
            pairs=pair_data.pairs,
            distances=pair_data.distances,
            categories=pair_data.categories,
        )

    hyperbolic_cfg = HyperbolicConfig(
        embedding_dim=args.embedding_dim,
        max_outer_steps=args.max_outer_steps,
        inner_steps=args.inner_steps,
        batch_size=args.batch_size,
        embedding_lr=args.embedding_lr,
        curvature_lr=args.curvature_lr,
        device=device,
    )
    estimator = HyperbolicCurvatureEstimator(num_nodes=args.n_clusters, config=hyperbolic_cfg)
    fit_result = estimator.fit(pair_data)

    summary = {
        "env": args.env,
        "checkpoint": str(checkpoint_path),
        "agent": args.agent,
        "optimal_c": fit_result.optimal_c,
        "alpha": fit_result.alpha,
        "n_clusters": args.n_clusters,
        "num_pairs": int(pair_data.pairs.shape[0]),
        "coverage": {
            "states": int(coverage.features.shape[0]),
            "transitions": int(coverage.transitions.shape[0]),
            "seeds": args.coverage_seeds,
            "steps_per_seed": args.coverage_steps,
            "subsample_stride": args.subsample_stride,
        },
    }
    (run_dir / "curvature_summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "training_history.json").write_text(json.dumps(fit_result.history, indent=2))
    np.savez_compressed(
        run_dir / "best_state.npz",
        embeddings=fit_result.best_state["embeddings"],
        theta=fit_result.best_state["theta"],
    )

    print(
        f"[result] Optimal curvature for env={args.env} "
        f"using checkpoint={checkpoint_path.name}: c* = {fit_result.optimal_c:.6f}"
    )
    print(f"[result] Full summary written to {run_dir / 'curvature_summary.json'}")


if __name__ == "__main__":
    main()
