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
import yaml

from hydra import initialize_config_dir, compose

# --- CONFIG: Modify these as needed ---
CKPT = (
    # "exp_local/hyperbolic/ppo_sn/bigfish_gen/2025.11.06_080602/checkpoint-22937600.pt"
    # "exp_local/ppo_sn/bigfish_gen/2025.11.06_080610/checkpoint-22937600.pt"
    # "exp_local/hyperbolic/ppo_sn/caveflyer_gen/2025.11.07_012323/checkpoint-22937600.pt"
    # "exp_local/ppo_sn/caveflyer_gen/2025.11.06_180159/checkpoint-22937600.pt"
    # "exp_local/hyperbolic/ppo_sn/dodgeball_gen/2025.11.06_110816/checkpoint-22937600.pt"
    # "exp_local/ppo_sn/dodgeball_gen/2025.11.06_095342/checkpoint-22937600.pt"
    # "exp_local/hyperbolic/ppo_sn/jumper_gen/2025.11.06_135738/checkpoint-22937600.pt"
    # "exp_local/ppo_sn/jumper_gen/2025.11.06_115254/checkpoint-22937600.pt"
    # "exp_local/hyperbolic/ppo_sn/miner_gen/2025.11.06_164736/checkpoint-22937600.pt"
    # "exp_local/ppo_sn/miner_gen/2025.11.06_135451/checkpoint-22937600.pt"
    # "exp_local/hyperbolic/ppo_sn/ninja_gen/2025.11.06_205244/checkpoint-22937600.pt"
    "exp_local/ppo_sn/ninja_gen/2025.11.06_160258/checkpoint-22937600.pt"
)
INFER_FROM_RUN = True                    # infer agent/env from run's .hydra/config.yaml
AGENT_CFG = "onpolicy/ppo"               # fallback if inference fails
ENV_CFG = "gen/ninja"                  # fallback env (e.g., gen/caveflyer)
N_EPISODES = 30                           # match training logs by default
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

    # Try to infer agent/env groups from the run's saved config to avoid arch mismatches
    inferred_agent_cfg = None
    inferred_env_cfg = None
    if INFER_FROM_RUN:
        run_cfg = ckpt_path.parent / ".hydra" / "config.yaml"
        if run_cfg.exists():
            try:
                with open(run_cfg, "r") as f:
                    cfg_y = yaml.safe_load(f)
                agent_name = str(cfg_y.get("agent_name", ""))
                env_name = str(cfg_y.get("env_name", ""))
                # Map saved agent_name to our Hydra group path
                if "hyperbolic" in agent_name:
                    inferred_agent_cfg = "onpolicy/hyperbolic/ppo"
                elif agent_name == "ppo_sn":
                    inferred_agent_cfg = "onpolicy/ppo_sn"
                elif agent_name:
                    inferred_agent_cfg = "onpolicy/ppo"
                if env_name:
                    inferred_env_cfg = f"gen/{env_name}"
            except Exception as e:
                print(f"Warning: failed to infer config from {run_cfg}: {e}")

    # Use absolute config directory so the script works from any CWD
    cfg_dir = str((Path(__file__).parent / "cfgs").resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        overrides = [
            f"agent@_global_={inferred_agent_cfg or AGENT_CFG}",
            f"env@_global_={inferred_env_cfg or ENV_CFG}",
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
