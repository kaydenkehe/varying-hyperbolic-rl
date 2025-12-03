#!/usr/bin/env python3
"""
Render a trained run to an MP4 video.

Usage:
  python render_video.py --run-dir <path to run dir> [--ckpt <checkpoint.pt>]
                         [--out <out.mp4>] [--episodes 1] [--fps 15]
                         [--det] [--cpu]

Notes:
- If --ckpt is omitted, the latest checkpoint in the run directory is used,
  preferring names starting with "checkpoint-229".
- The run directory should contain a saved Hydra config at .hydra/config.yaml
  (as produced by main_hydra.py).
- The video is written to results/videos/<env>_c<curv>.mp4 by default.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Sequence

import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf

from main_hydra import make_models


_CKPT_RE = re.compile(r"checkpoint-(\d+)\.pt$")


def _find_latest_checkpoint(run_dir: Path) -> Path:
    ckpts = list(run_dir.rglob("checkpoint-*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-*.pt found under {run_dir}")
    preferred = [p for p in ckpts if p.name.startswith("checkpoint-229")]
    pool = preferred if preferred else ckpts

    def step_num(p: Path) -> int:
        m = _CKPT_RE.search(p.name)
        return int(m.group(1)) if m else -1

    pool.sort(key=step_num, reverse=True)
    return pool[0]


def _load_cfg(run_dir: Path, force_cpu: bool, episodes: int) -> OmegaConf:
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing Hydra config: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    # Apply lightweight overrides without resolving all interpolations.
    cfg.disable_cuda = bool(force_cpu)
    cfg.n_eval_envs = int(episodes)
    # For video, force single env and rgb_array rendering
    cfg.env.num = 1
    cfg.eval_env.num = 1
    cfg.eval_env.render_mode = "rgb_array"
    return cfg


def _frames_to_video(frames: Sequence[np.ndarray], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Prefer mp4/ffmpeg when available
        with imageio.get_writer(out_path, fps=fps, codec="h264") as writer:
            for frame in frames:
                writer.append_data(frame)
    except TypeError as e:
        # Fallback: write a GIF instead if ffmpeg/imageio-ffmpeg are incompatible.
        alt_path = out_path.with_suffix(".gif")
        imageio.mimsave(alt_path, list(frames), fps=fps)
        print(
            f"Warning: failed to write MP4 via ffmpeg ({e}); "
            f"wrote GIF instead at {alt_path}"
        )


def render_episode(run_dir: Path, ckpt_path: Optional[Path], out_path: Optional[Path], episodes: int, fps: int, det: bool, force_cpu: bool) -> Path:
    if ckpt_path is None:
        ckpt_path = _find_latest_checkpoint(run_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = _load_cfg(run_dir, force_cpu=force_cpu, episodes=episodes)
    agent, buffer, env, tester, preproc = make_models(cfg)
    loaded_preproc = agent.load(str(ckpt_path))
    if loaded_preproc is not None:
        preproc = loaded_preproc
    # Ensure preprocessor uses the same device as the agent/model
    if hasattr(cfg, "device"):
        try:
            preproc.device = cfg.device
        except Exception:
            pass

    # Fresh eval env with rendering
    import hydra

    eval_env = hydra.utils.instantiate(cfg.eval_env)
    frames = []

    for ep in range(episodes):
        rew, obs, first = eval_env.observe()
        done = False
        last_returns = 0.0
        while not done:
            frames.append(obs["rgb"][0])
            obs_proc = preproc.preprocess_obs((obs["rgb"].transpose(0, 3, 1, 2)).astype(np.float32))
            act = agent.act(obs_proc, det=det)
            eval_env.act(act)
            rew, obs, first = eval_env.observe()
            last_returns = rew + last_returns
            done = bool(first[0])
        # append final frame
        frames.append(obs["rgb"][0])

    env_name = getattr(cfg, "env_name", "env")
    label = f"c{cfg.curvature}"
    if out_path is None:
        out_path = Path("results") / "videos" / f"{env_name}_{label}.mp4"

    _frames_to_video(frames, out_path, fps)
    print(f"Wrote video to {out_path} (frames={len(frames)}, episodes={episodes}, ckpt={ckpt_path.name})")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a trained hyperbolic model to video.")
    p.add_argument("--run-dir", required=True, type=Path, help="Run directory containing .hydra/config.yaml and checkpoints.")
    p.add_argument("--ckpt", type=Path, help="Optional checkpoint path (default: latest in run dir).")
    p.add_argument("--out", type=Path, help="Output video path (default: results/videos/<env>_c<curv>.mp4).")
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to render (default: 1).")
    p.add_argument("--fps", type=int, default=15, help="Output video FPS (default: 15).")
    p.add_argument("--det", action="store_true", help="Use deterministic policy actions.")
    p.add_argument("--cpu", action="store_true", help="Force CPU for rendering.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    render_episode(
        run_dir=args.run_dir,
        ckpt_path=args.ckpt,
        out_path=args.out,
        episodes=args.episodes,
        fps=args.fps,
        det=args.det,
        force_cpu=args.cpu,
    )


if __name__ == "__main__":
    main()
