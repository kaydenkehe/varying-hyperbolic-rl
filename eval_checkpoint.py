#!/usr/bin/env python3
"""
Evaluate a saved checkpoint and report mean/std of rewards over N runs.

Edit the CONFIG section below to point to your checkpoint and config.
"""

from __future__ import annotations

import sys
from statistics import mean, pstdev
from pathlib import Path
from typing import Sequence

from hydra import initialize_config_dir, compose

# --- CONFIG: Modify these as needed ---
CKPT = (
    "exp_local/hyperbolic/ppo_sn/bigfish_gen/2025.11.06_080602/checkpoint-9830400.pt"
)
AGENT_CFG = "onpolicy/hyperbolic/ppo"  # or "onpolicy/ppo"
ENV_CFG = "gen/bigfish"                 # e.g., gen/caveflyer, gen/ninja
N_EPISODES = 10                          # match training logs by default
DET = False                              # match training (stochastic) by default
DISABLE_CUDA = False                     # set True to force CPU
# Optional extra Hydra overrides, e.g. to fix seeds/levels
# Example: EXTRA_OVERRIDES = ["eval_env.start_level=0", "eval_env.num_levels=200"]
EXTRA_OVERRIDES = []
# --------------------------------------


def _summarize(returns: Sequence[float]) -> tuple[float, float]:
    if not returns:
        return float("nan"), float("nan")
    # Use population std for a stable estimate across fixed N episodes
    mu = mean(returns)
    sd = pstdev(returns) if len(returns) > 1 else 0.0
    return mu, sd


def main() -> None:
    # Late imports to avoid importing project modules before Hydra config
    from main_hydra import make_models

    ckpt_path = Path(CKPT)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Use absolute config directory so the script works from any CWD
    cfg_dir = str((Path(__file__).parent / "cfgs").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        overrides = [
            f"agent={AGENT_CFG}",
            f"env={ENV_CFG}",
            f"n_eval_envs={N_EPISODES}",  # tester.min_eval_episodes mirrors this
            f"disable_cuda={'true' if DISABLE_CUDA else 'false'}",
        ] + list(EXTRA_OVERRIDES)

        cfg = compose(
            config_name="config",
            overrides=overrides,
        )

    agent, buffer, env, tester, preproc = make_models(cfg)

    # Load checkpoint; may return a stored preprocessor
    loaded_preproc = agent.load(str(ckpt_path))
    if loaded_preproc is not None:
        tester.preprocessor = loaded_preproc

    rets = tester.evaluate(agent, det=DET)
    # When train_env_cfg is present, evaluate() returns both test and train
    if isinstance(rets, dict):
        rets = rets.get("", rets)  # prefer test returns under ""

    mu, sd = _summarize(rets)
    print("episodes:", len(rets))
    print("returns:", [round(float(r), 3) for r in rets])
    print("mean:", round(mu, 3))
    print("std:", round(sd, 3))


if __name__ == "__main__":
    main()
