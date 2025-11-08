#!/usr/bin/env python3
"""
Evaluate one or more saved checkpoints and aggregate results.

Updates:
- Runs all 12 provided PPO / Hyperbolic PPO checkpoints (6 tasks x 2 methods).
- Bumps default N (episodes) to 100 and makes it easy to change via CLI.
- Automatically writes aggregated results to results/results.csv with columns:
  Task, PPO Mean, PPO Std, Hyperbolic PPO Mean, Hyperbolic PPO Std, Percent Change
"""

from __future__ import annotations

import sys
from statistics import mean, pstdev
from pathlib import Path
from typing import Sequence, Dict, Tuple, List
import yaml

from hydra import initialize_config_dir, compose

# --- CONFIG: Modify these as needed ---
# Checkpoints for 6 Procgen tasks x {PPO, Hyperbolic PPO}
# Paths are based on the commented examples previously in this file.
MODELS: Dict[str, Dict[str, str]] = {
    "bigfish": {
        "hppo": "exp_local/hyperbolic/ppo_sn/bigfish_gen/2025.11.06_080602/checkpoint-22937600.pt",
        "ppo":  "exp_local/ppo_sn/bigfish_gen/2025.11.06_080610/checkpoint-22937600.pt",
    },
    "caveflyer": {
        "hppo": "exp_local/hyperbolic/ppo_sn/caveflyer_gen/2025.11.07_012323/checkpoint-22937600.pt",
        "ppo":  "exp_local/ppo_sn/caveflyer_gen/2025.11.06_180159/checkpoint-22937600.pt",
    },
    "dodgeball": {
        "hppo": "exp_local/hyperbolic/ppo_sn/dodgeball_gen/2025.11.06_110816/checkpoint-22937600.pt",
        "ppo":  "exp_local/ppo_sn/dodgeball_gen/2025.11.06_095342/checkpoint-22937600.pt",
    },
    "jumper": {
        "hppo": "exp_local/hyperbolic/ppo_sn/jumper_gen/2025.11.06_135738/checkpoint-22937600.pt",
        "ppo":  "exp_local/ppo_sn/jumper_gen/2025.11.06_115254/checkpoint-22937600.pt",
    },
    "miner": {
        "hppo": "exp_local/hyperbolic/ppo_sn/miner_gen/2025.11.06_164736/checkpoint-22937600.pt",
        "ppo":  "exp_local/ppo_sn/miner_gen/2025.11.06_135451/checkpoint-22937600.pt",
    },
    "ninja": {
        "hppo": "exp_local/hyperbolic/ppo_sn/ninja_gen/2025.11.06_205244/checkpoint-22937600.pt",
        "ppo":  "exp_local/ppo_sn/ninja_gen/2025.11.06_160258/checkpoint-22937600.pt",
    },
}
INFER_FROM_RUN = True                    # infer agent/env from run's .hydra/config.yaml
AGENT_CFG = "onpolicy/ppo"               # fallback if inference fails
ENV_CFG = "gen/ninja"                    # fallback env (e.g., gen/caveflyer)
N_EPISODES = 100                          # default evaluation episodes (can override with --episodes)
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


def _infer_from_run(ckpt_path: Path) -> Tuple[str | None, str | None, str | None]:
    """Infer agent/env configs and task name from the run's saved Hydra config.

    Returns (agent_cfg, env_cfg, task_name). Any item may be None on failure.
    """
    run_cfg = ckpt_path.parent / ".hydra" / "config.yaml"
    if not run_cfg.exists():
        return None, None, None
    try:
        with open(run_cfg, "r") as f:
            cfg_y = yaml.safe_load(f)
        agent_name = str(cfg_y.get("agent_name", ""))
        env_name = str(cfg_y.get("env_name", ""))
        if "hyperbolic" in agent_name:
            agent_cfg = "onpolicy/hyperbolic/ppo"
        elif agent_name == "ppo_sn":
            agent_cfg = "onpolicy/ppo_sn"
        elif agent_name:
            agent_cfg = "onpolicy/ppo"
        else:
            agent_cfg = None
        env_cfg = f"gen/{env_name}" if env_name else None
        task_name = env_name or None
        return agent_cfg, env_cfg, task_name
    except Exception as e:
        print(f"Warning: failed to infer config from {run_cfg}: {e}")
        return None, None, None


def _evaluate_one(ckpt_path: Path, n_episodes: int) -> Tuple[str, str, float, float]:
    """Evaluate a single checkpoint and return (task, method, mean, std).

    method in {"ppo", "hppo"}.
    """
    # Late import inside Hydra context in main(), but type here for clarity
    # from main_hydra import make_models

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    inferred_agent_cfg, inferred_env_cfg, task_name = (None, None, None)
    agent_cfg = None
    env_cfg = None
    if INFER_FROM_RUN:
        inferred_agent_cfg, inferred_env_cfg, task_name = _infer_from_run(ckpt_path)
        agent_cfg = inferred_agent_cfg or AGENT_CFG
        env_cfg = inferred_env_cfg or ENV_CFG
    else:
        agent_cfg = AGENT_CFG
        env_cfg = ENV_CFG

    # Determine method label for reporting
    method = "hppo" if (inferred_agent_cfg and "hyperbolic" in inferred_agent_cfg) else "ppo"
    if task_name is None:
        # Fall back to extracting from path chunk like .../<task>_gen/...
        parts = [p for p in ckpt_path.parts if p.endswith("_gen")]
        if parts:
            task_name = parts[0].replace("_gen", "")
        else:
            task_name = "unknown"

    overrides = [
        f"agent@_global_={agent_cfg}",
        f"env@_global_={env_cfg}",
        f"n_eval_envs={n_episodes}",  # tester.min_eval_episodes mirrors this
        f"disable_cuda={'true' if DISABLE_CUDA else 'false'}",
    ] + list(EXTRA_OVERRIDES)

    cfg = compose(config_name="config", overrides=overrides)

    # import after compose
    from main_hydra import make_models
    agent, buffer, env, tester, preproc = make_models(cfg)

    loaded_preproc = agent.load(str(ckpt_path))
    if loaded_preproc is not None:
        tester.preprocessor = loaded_preproc

    rets = tester.evaluate(agent, det=DET)
    if isinstance(rets, dict):
        rets = rets.get("", rets)
    mu, sd = _summarize(rets)
    return task_name, method, mu, sd


def _write_results_csv(rows: List[Tuple[str, float, float, float, float]], out_path: Path) -> None:
    """Write aggregated results to CSV including percent change."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Header
    lines = [
        "Task,PPO Mean,PPO Std,Hyperbolic PPO Mean,Hyperbolic PPO Std,Percent Change"
    ]
    # rows tuple: (task, ppo_mu, ppo_sd, hppo_mu, hppo_sd)
    for task, ppo_mu, ppo_sd, hppo_mu, hppo_sd in rows:
        # Percent change: (hppo - ppo) / ppo * 100
        if ppo_mu == 0:
            pct = float("nan")
        else:
            pct = (hppo_mu - ppo_mu) / ppo_mu * 100.0
        lines.append(
            f"{task},{ppo_mu:.2f},{ppo_sd:.2f},{hppo_mu:.2f},{hppo_sd:.2f},{pct:.2f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    # Parse simple CLI override: --episodes N
    global N_EPISODES
    args = sys.argv[1:]
    if "--episodes" in args:
        try:
            idx = args.index("--episodes")
            N_EPISODES = int(args[idx + 1])
        except Exception as e:
            print(f"Warning: failed to parse --episodes: {e}")

    # Use absolute config directory so the script works from any CWD
    cfg_dir = str((Path(__file__).parent / "cfgs").resolve())
    results: Dict[str, Dict[str, Tuple[float, float]]] = {}
    tasks_in_order = ["bigfish", "caveflyer", "dodgeball", "jumper", "miner", "ninja"]

    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        # Evaluate all checkpoints
        for task, methods in MODELS.items():
            for method_key in ("ppo", "hppo"):
                ckpt = methods.get(method_key)
                if not ckpt:
                    continue
                ckpt_path = Path(ckpt)
                try:
                    tname, method, mu, sd = _evaluate_one(ckpt_path, N_EPISODES)
                    results.setdefault(tname, {})[method] = (mu, sd)
                    print(f"Evaluated {tname} [{method}] => mean={mu:.3f} sd={sd:.3f}")
                except Exception as e:
                    print(f"Error evaluating {ckpt_path}: {e}")

    # Construct rows in a stable task order
    rows: List[Tuple[str, float, float, float, float]] = []
    for task in tasks_in_order:
        rec = results.get(task, {})
        ppo_mu, ppo_sd = rec.get("ppo", (float("nan"), float("nan")))
        hppo_mu, hppo_sd = rec.get("hppo", (float("nan"), float("nan")))
        rows.append((task, ppo_mu, ppo_sd, hppo_mu, hppo_sd))

    out_csv = Path("results") / "results.csv"
    _write_results_csv(rows, out_csv)
    print(f"Wrote aggregated CSV to: {out_csv}")


if __name__ == "__main__":
    main()
