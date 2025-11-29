#!/usr/bin/env python3
"""
Evaluate latest checkpoints for a list of curvature settings and write
per-curvature result tables under results/curv<curv>/results.csv.

Defaults:
- Curvature list: 0.1, 0.5, 1.0, 2.0, 5.0 (skip c=0; base results.csv already
  has Euclidean/Hyperbolic columns).
- Each curvature expects a directory results/curv<curv>/ with env subdirs
  containing checkpoints; the latest checkpoint is chosen by step, preferring
  names starting with "checkpoint-229" when present.
- Uses each run's saved .hydra/config.yaml to build the model so curvature and
  other overrides are preserved during evaluation.
- Base table: results/results.csv. This script drops the Percent Change column
  (if present) and appends two columns for the current curvature: "c=<curv>
  Mean", "c=<curv> Std".

Notes:
- Outputs one CSV per curvature: results/curv<curv>/results.csv.
- Also emits a consolidated sweep CSV at results/curvature_sweep.csv if at
  least one curvature produces results.
- Curvature directories are not created automatically; missing dirs are
  skipped with a warning.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from omegaconf import OmegaConf

from main_hydra import make_models


@dataclass
class SweepConfig:
    curvatures: List[float] = field(
        default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0]
    )
    results_root: Path = Path("results")
    base_csv: Path = Path("results/results.csv")
    episodes: int = 100
    det: bool = False
    force_cpu: bool = False
    prefer_prefix: str = "checkpoint-229"
    summary_csv: Path = Path("results/curvature_sweep.csv")


def _summarize(returns: Sequence[float]) -> Tuple[float, float]:
    if not returns:
        return float("nan"), float("nan")
    m = sum(returns) / len(returns)
    if len(returns) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in returns) / len(returns)
    return m, math.sqrt(var)


_CKPT_RE = re.compile(r"checkpoint-(\d+)\.pt$")


def _find_latest_checkpoint(env_dir: Path, prefer_prefix: str) -> Optional[Path]:
    candidates = list(env_dir.rglob("checkpoint-*.pt"))
    if not candidates:
        return None

    def step_num(p: Path) -> int:
        m = _CKPT_RE.search(p.name)
        return int(m.group(1)) if m else -1

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

    try:
        cfg = OmegaConf.create(OmegaConf.to_container(run_cfg, resolve=True))
    except Exception:
        cfg = run_cfg

    cfg["n_eval_envs"] = int(episodes)
    cfg["disable_cuda"] = bool(force_cpu)

    agent, buffer, env, tester, preproc = make_models(cfg)

    loaded_preproc = agent.load(str(ckpt))
    if loaded_preproc is not None:
        tester.preprocessor = loaded_preproc

    returns = tester.evaluate(agent, det=det)
    if isinstance(returns, dict):
        returns = returns.get("", returns)
    return _summarize(returns)


def _read_base_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")
    return rows[0], rows[1:]


def _write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _format_curv_label(curv: float) -> str:
    # Trim trailing zeros for readable folder/column names
    text = ("%g" % curv).rstrip("0").rstrip(".")
    return text or "0"


def _env_dirs(root: Path) -> List[Path]:
    dirs = []
    if root.is_dir():
        for p in root.iterdir():
            if p.is_dir() and (p / ".hydra" / "config.yaml").exists():
                dirs.append(p)
    return sorted(dirs)


def _drop_percent_change(header: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    percent_idx: Optional[int] = None
    for i, h in enumerate(header):
        if h.strip().lower() == "percent change":
            percent_idx = i
            break
    if percent_idx is None:
        return header, rows
    new_header = header[:percent_idx] + header[percent_idx + 1 :]
    new_rows = [r[:percent_idx] + r[percent_idx + 1 :] if len(r) > percent_idx else r for r in rows]
    return new_header, new_rows


def _extract_hppo_scores(header: List[str], rows: List[List[str]]) -> Dict[str, Tuple[float, float]]:
    mean_idx = std_idx = None
    header_l = [h.strip().lower() for h in header]
    for i, h in enumerate(header_l):
        if h == "hyperbolic ppo mean":
            mean_idx = i
        elif h == "hyperbolic ppo std":
            std_idx = i
    if mean_idx is None or std_idx is None:
        return {}
    scores: Dict[str, Tuple[float, float]] = {}
    for row in rows:
        if len(row) <= max(mean_idx, std_idx):
            continue
        task = row[0]
        try:
            mu = float(row[mean_idx])
        except Exception:
            mu = float("nan")
        try:
            sd = float(row[std_idx])
        except Exception:
            sd = float("nan")
        scores[task] = (mu, sd)
    return scores


def evaluate_curvature(
    curv: float,
    cfg: SweepConfig,
    base_header: List[str],
    base_rows: List[List[str]],
    hppo_scores: Dict[str, Tuple[float, float]],
) -> Tuple[str, Dict[str, Tuple[float, float]], Path]:
    label = _format_curv_label(curv)
    curv_dir = cfg.results_root / f"curv{label}"

    # Special-case c=1: copy from base Hyperbolic PPO columns instead of re-evaluating.
    if math.isclose(curv, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        scores = hppo_scores
        header = list(base_header) + [f"c={label} Mean", f"c={label} Std"]
        rows: List[List[str]] = []
        for row in base_rows:
            if not row:
                continue
            task = row[0]
            mu, sd = scores.get(task, (float("nan"), float("nan")))
            mu_s = f"{mu:.2f}" if math.isfinite(mu) else "nan"
            sd_s = f"{sd:.2f}" if math.isfinite(sd) else "nan"
            rows.append(row + [mu_s, sd_s])
        out_path = curv_dir / "results.csv"
        _write_csv(out_path, header, rows)
        print(f"Copied c={label} from base results into: {out_path}")
        return label, scores, out_path

    if not curv_dir.exists():
        print(f"Curvature dir missing for c={label}: {curv_dir} (skipping)")
        return label, {}, curv_dir / "results.csv"

    env_dirs = _env_dirs(curv_dir)
    scores: Dict[str, Tuple[float, float]] = {}
    for env_dir in env_dirs:
        env_name = env_dir.name
        ckpt = _find_latest_checkpoint(env_dir, cfg.prefer_prefix)
        if ckpt is None:
            print(f"No checkpoint found in {env_dir} (c={label}); marking NaN.")
            scores[env_name] = (float("nan"), float("nan"))
            continue
        try:
            mu, sd = _evaluate_ckpt(env_dir, ckpt, cfg.episodes, cfg.det, cfg.force_cpu)
            print(f"Evaluated c={label} {env_name} @ {ckpt.name}: mean={mu:.3f} std={sd:.3f}")
            scores[env_name] = (mu, sd)
        except Exception as e:
            print(f"Error evaluating c={label} {env_name} ({ckpt}): {e}")
            scores[env_name] = (float("nan"), float("nan"))

    header = list(base_header) + [f"c={label} Mean", f"c={label} Std"]
    rows: List[List[str]] = []
    for row in base_rows:
        if not row:
            continue
        task = row[0]
        mu, sd = scores.get(task, (float("nan"), float("nan")))
        mu_s = f"{mu:.2f}" if math.isfinite(mu) else "nan"
        sd_s = f"{sd:.2f}" if math.isfinite(sd) else "nan"
        rows.append(row + [mu_s, sd_s])

    out_path = curv_dir / "results.csv"
    _write_csv(out_path, header, rows)
    print(f"Wrote c={label} table to: {out_path}")
    return label, scores, out_path


def run(cfg: SweepConfig) -> None:
    if not cfg.base_csv.exists():
        raise FileNotFoundError(f"Base CSV not found: {cfg.base_csv}")

    base_header, base_rows = _read_base_csv(cfg.base_csv)
    base_header, base_rows = _drop_percent_change(base_header, base_rows)
    hppo_scores = _extract_hppo_scores(base_header, base_rows)

    sweep_scores: Dict[str, Dict[str, Tuple[float, float]]] = {}
    written_any = False
    for curv in cfg.curvatures:
        label, scores, out_path = evaluate_curvature(curv, cfg, base_header, base_rows, hppo_scores)
        if scores:
            sweep_scores[label] = scores
            written_any = True

    if not written_any:
        print("No curvature results were written (missing dirs or checkpoints).")
        return

    # Build consolidated sweep CSV
    sweep_header = list(base_header)
    labels_in_order = [_format_curv_label(c) for c in cfg.curvatures if _format_curv_label(c) in sweep_scores]
    for label in labels_in_order:
        sweep_header += [f"c={label} Mean", f"c={label} Std"]

    sweep_rows: List[List[str]] = []
    for row in base_rows:
        if not row:
            continue
        task = row[0]
        enriched = list(row)
        for label in labels_in_order:
            mu, sd = sweep_scores[label].get(task, (float("nan"), float("nan")))
            mu_s = f"{mu:.2f}" if math.isfinite(mu) else "nan"
            sd_s = f"{sd:.2f}" if math.isfinite(sd) else "nan"
            enriched += [mu_s, sd_s]
        sweep_rows.append(enriched)

    _write_csv(cfg.summary_csv, sweep_header, sweep_rows)
    print(f"Wrote consolidated sweep CSV to: {cfg.summary_csv}")


def _parse_args(argv: List[str]) -> SweepConfig:
    cfg = SweepConfig()
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-c", "--curvatures"):
            i += 1
            cfg.curvatures = [float(x) for x in argv[i].split(",") if x]
        elif arg in ("-n", "--episodes"):
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
        elif arg in ("-s", "--summary-csv"):
            i += 1
            cfg.summary_csv = Path(argv[i])
        elif arg == "--prefer-prefix":
            i += 1
            cfg.prefer_prefix = argv[i]
        else:
            print(f"Warning: unknown arg {arg} (ignored)")
        i += 1
    return cfg


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    run(args)
