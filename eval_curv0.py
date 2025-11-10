#!/usr/bin/env python3
"""
Evaluate the latest c=0 checkpoints under results/curv0 and append their
mean/std to a copy of results/results.csv.

Behavior:
- Looks for subdirectories in results/curv0 named by environment (e.g., ninja).
- In each env dir, finds the latest checkpoint matching "checkpoint-229*.pt";
  if none match that prefix, falls back to the numerically largest
  "checkpoint-*.pt".
- Loads the saved Hydra config from <env>/.hydra/config.yaml to instantiate
  the model/env, overrides n_eval_envs (episodes) and disable_cuda as needed,
  then evaluates to compute mean/std returns.
- Writes results/results_with_c0.csv which is results/results.csv with two
  extra columns appended: "c=0 Mean","c=0 Std" for each Task row.

Notes:
- This script does not modify results/results.csv; it writes a new file.
- If a checkpoint or evaluation is missing/fails for an env, the c=0 columns
  will be NaN for that Task.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from hydra.utils import instantiate
from omegaconf import OmegaConf

# Local imports
from main_hydra import make_models


@dataclass
class EvalConfig:
    curv0_root: Path = Path("results/curv0")
    base_csv: Path = Path("results/results.csv")
    out_csv: Path = Path("results/results_with_c0.csv")
    episodes: int = 100  # default evaluation episodes
    det: bool = False    # stochastic policy by default (match training)
    force_cpu: bool = False
    prefer_prefix: str = "checkpoint-229"  # prioritize final checkpoints


def _summarize(returns: Sequence[float]) -> Tuple[float, float]:
    if not returns:
        return float("nan"), float("nan")
    # population std for consistent comparison
    m = sum(returns) / len(returns)
    if len(returns) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in returns) / len(returns)
    return m, math.sqrt(var)


_CKPT_RE = re.compile(r"checkpoint-(\d+)\.pt$")


def _find_latest_checkpoint(env_dir: Path, prefer_prefix: str) -> Optional[Path]:
    if not env_dir.is_dir():
        return None
    candidates = list(env_dir.rglob("checkpoint-*.pt"))
    if not candidates:
        return None

    def step_num(p: Path) -> int:
        m = _CKPT_RE.search(p.name)
        return int(m.group(1)) if m else -1

    # Prefer checkpoints that start with the given prefix, if any exist
    preferred = [p for p in candidates if p.name.startswith(prefer_prefix)]
    pool = preferred if preferred else candidates
    pool.sort(key=step_num, reverse=True)
    return pool[0]


def _load_run_cfg(env_dir: Path) -> Optional[OmegaConf]:
    cfg_path = env_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        return OmegaConf.load(cfg_path)
    except Exception as e:
        print(f"Warning: failed to load {cfg_path}: {e}")
        return None


def _evaluate_ckpt(env_dir: Path, ckpt: Path, episodes: int, det: bool, force_cpu: bool) -> Tuple[float, float]:
    run_cfg = _load_run_cfg(env_dir)
    if run_cfg is None:
        raise RuntimeError(f"Missing Hydra config in {env_dir}/.hydra/config.yaml")

    # Apply evaluation-time overrides
    try:
        # Ensure interpolation is resolved before updates
        cfg = OmegaConf.create(OmegaConf.to_container(run_cfg, resolve=True))
    except Exception:
        cfg = run_cfg

    # Episodes control both eval env count and tester min episodes in this repo
    cfg["n_eval_envs"] = int(episodes)
    cfg["disable_cuda"] = bool(force_cpu)

    # Instantiate models/env/tester from saved config
    agent, buffer, env, tester, preproc = make_models(cfg)

    loaded_preproc = agent.load(str(ckpt))
    if loaded_preproc is not None:
        tester.preprocessor = loaded_preproc

    returns = tester.evaluate(agent, det=det)
    if isinstance(returns, dict):
        returns = returns.get("", returns)
    mu, sd = _summarize(returns)
    return mu, sd


def _read_base_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")
    header, data = rows[0], rows[1:]
    return header, data


def _write_augmented_csv(path: Path, header: List[str], data: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


def run(cfg: EvalConfig) -> None:
    if not cfg.curv0_root.exists():
        raise FileNotFoundError(f"Directory not found: {cfg.curv0_root}")
    if not cfg.base_csv.exists():
        raise FileNotFoundError(f"Base CSV not found: {cfg.base_csv}")

    # Collect env dirs
    env_dirs = [p for p in cfg.curv0_root.iterdir() if p.is_dir()]
    env_names = {p.name for p in env_dirs}

    # Evaluate each env
    c0_scores: Dict[str, Tuple[float, float]] = {}
    for env_dir in sorted(env_dirs):
        env_name = env_dir.name
        ckpt = _find_latest_checkpoint(env_dir, cfg.prefer_prefix)
        if ckpt is None:
            print(f"No checkpoint found for {env_name} in {env_dir}; skipping.")
            c0_scores[env_name] = (float("nan"), float("nan"))
            continue
        try:
            mu, sd = _evaluate_ckpt(env_dir, ckpt, cfg.episodes, cfg.det, cfg.force_cpu)
            print(f"Evaluated {env_name} @ {ckpt.name}: mean={mu:.3f} std={sd:.3f}")
            c0_scores[env_name] = (mu, sd)
        except Exception as e:
            print(f"Error evaluating {env_name} ({ckpt}): {e}")
            c0_scores[env_name] = (float("nan"), float("nan"))

    # Load base CSV and append c=0 columns
    base_header, base_rows = _read_base_csv(cfg.base_csv)
    out_header = list(base_header) + ["c=0 Mean", "c=0 Std"]

    out_rows: List[List[str]] = []
    for row in base_rows:
        if not row:
            continue
        task = row[0]
        mu, sd = c0_scores.get(task, (float("nan"), float("nan")))
        mu_s = f"{mu:.2f}" if math.isfinite(mu) else "nan"
        sd_s = f"{sd:.2f}" if math.isfinite(sd) else "nan"
        out_rows.append(row + [mu_s, sd_s])

    _write_augmented_csv(cfg.out_csv, out_header, out_rows)
    print(f"Wrote augmented CSV to: {cfg.out_csv}")


def _parse_args(argv: List[str]) -> EvalConfig:
    cfg = EvalConfig()
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-n", "--episodes"):
            i += 1
            cfg.episodes = int(argv[i])
        elif arg in ("--det",):
            cfg.det = True
        elif arg in ("--cpu", "--force-cpu"):
            cfg.force_cpu = True
        elif arg in ("--curv0", "--curv0-root"):
            i += 1
            cfg.curv0_root = Path(argv[i])
        elif arg in ("-i", "--base-csv"):
            i += 1
            cfg.base_csv = Path(argv[i])
        elif arg in ("-o", "--out-csv"):
            i += 1
            cfg.out_csv = Path(argv[i])
        elif arg in ("--prefer-prefix",):
            i += 1
            cfg.prefer_prefix = argv[i]
        else:
            print(f"Warning: unknown arg {arg} (ignored)")
        i += 1
    return cfg


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    run(args)

