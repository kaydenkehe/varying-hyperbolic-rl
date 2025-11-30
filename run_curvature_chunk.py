#!/usr/bin/env python3
"""Run training/eval for one curvature/env chunk (1–10).

Each chunk ID corresponds to a curvature and half of the 8 Procgen
environments used in results/results.csv:

Curvatures (fixed order): [0.1, 0.5, 1.0, 2.0, 5.0]

Chunk mapping (assuming 8 tasks in results/results.csv):
  1 → c=0.1, first  4 envs
  2 → c=0.1, second 4 envs
  3 → c=0.5, first  4 envs
  4 → c=0.5, second 4 envs
  5 → c=1.0, first  4 envs (no training; c=1 copied from base table)
  6 → c=1.0, second 4 envs (no training)
  7 → c=2.0, first  4 envs
  8 → c=2.0, second 4 envs
  9 → c=5.0, first  4 envs
 10 → c=5.0, second 4 envs

For c≠1, the script:
- Ensures a run exists under results/curv<curv>/<env>/<timestamp>/
  (launching training via main_hydra.py if needed), and
- Evaluates the latest checkpoint for the requested number of episodes,
  printing mean/std for each env.

It does NOT write the aggregated CSVs; after all chunks finish, run
eval_curvature_sweep.py (with --no-train) to build per-curvature and
consolidated results tables.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from eval_curvature_sweep import (
    _best_run_dir,
    _drop_percent_change,
    _env_names,
    _evaluate_ckpt,
    _format_curv_label,
    _launch_training,
    _read_base_csv,
)


@dataclass
class ChunkConfig:
    chunk_id: int
    episodes: int = 1000
    det: bool = False
    force_cpu: bool = False
    # Match eval_curvature_sweep.py: write new runs under results_par
    results_root: Path = Path("results_par")
    base_csv: Path = Path("results/results.csv")
    prefer_prefix: str = "checkpoint-229"
    train_if_missing: bool = True


CURVATURES: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0]


def _parse_args(argv: List[str]) -> ChunkConfig:
    if not argv:
        raise SystemExit("Usage: run_curvature_chunk.py <chunk_id 1-10> [options]")
    try:
        chunk_id = int(argv[0])
    except ValueError:
        raise SystemExit("First argument must be an integer chunk ID in [1,10]")
    if not (1 <= chunk_id <= 10):
        raise SystemExit("chunk_id must be between 1 and 10")

    cfg = ChunkConfig(chunk_id=chunk_id)
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("-n", "--episodes"):
            i += 1
            cfg.episodes = int(argv[i])
        elif arg == "--det":
            cfg.det = True
        elif arg in ("--cpu", "--force-cpu"):
            cfg.force_cpu = True
        elif arg == "--results-root":
            i += 1
            cfg.results_root = Path(argv[i])
        elif arg in ("-i", "--base-csv"):
            i += 1
            cfg.base_csv = Path(argv[i])
        elif arg == "--prefer-prefix":
            i += 1
            cfg.prefer_prefix = argv[i]
        elif arg == "--no-train":
            cfg.train_if_missing = False
        else:
            print(f"Warning: unknown arg {arg} (ignored)")
        i += 1
    return cfg


def _chunk_to_curv_and_half(chunk_id: int) -> tuple[float, int]:
    idx = chunk_id - 1
    curv_idx = idx // 2
    half_idx = idx % 2
    return CURVATURES[curv_idx], half_idx


def main(cfg: ChunkConfig) -> None:
    if not cfg.base_csv.exists():
        raise FileNotFoundError(f"Base CSV not found: {cfg.base_csv}")

    base_header, base_rows = _read_base_csv(cfg.base_csv)
    base_header, base_rows = _drop_percent_change(base_header, base_rows)
    env_names = _env_names(base_rows)
    if len(env_names) < 2:
        raise RuntimeError("Expected at least 2 tasks in results.csv to split.")

    curv, half_idx = _chunk_to_curv_and_half(cfg.chunk_id)
    label = _format_curv_label(curv)

    mid = len(env_names) // 2
    first_half = env_names[:mid]
    second_half = env_names[mid:]
    subset = first_half if half_idx == 0 else second_half
    print(f"Chunk {cfg.chunk_id}: c={label}, envs={subset}")

    # For c=1 we rely on the existing baseline; no new training is launched.
    if math.isclose(curv, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        print("c=1 chunk: no training/eval needed (baseline already in results.csv).")
        return

    curv_dir = cfg.results_root / f"curv{label}"
    curv_dir.mkdir(parents=True, exist_ok=True)

    for env_name in subset:
        env_root = curv_dir / env_name
        env_root.mkdir(parents=True, exist_ok=True)
        run_dir, ckpt = _best_run_dir(env_root, cfg.prefer_prefix)
        if ckpt is None:
            if cfg.train_if_missing:
                try:
                    run_dir = _launch_training(curv, env_name, env_root)
                except Exception as e:
                    print(f"[train] Failed for c={label} env={env_name}: {e}")
                    continue
                run_dir, ckpt = _best_run_dir(env_root, cfg.prefer_prefix)
            else:
                print(f"No checkpoint found for c={label} env={env_name}; skipping.")
                continue
        if ckpt is None or run_dir is None:
            print(f"No checkpoint available after training for c={label} env={env_name}.")
            continue
        try:
            mu, sd = _evaluate_ckpt(run_dir, ckpt, cfg.episodes, cfg.det, cfg.force_cpu)
            print(f"[eval] c={label} env={env_name}: mean={mu:.3f} std={sd:.3f} ({ckpt.name})")
        except Exception as e:
            print(f"[eval] Failed for c={label} env={env_name} ({ckpt}): {e}")


if __name__ == "__main__":
    cfg = _parse_args(sys.argv[1:])
    main(cfg)
