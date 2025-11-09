# Curvature Estimation Methodology

This document records how the implementation inside `curvature/` realizes the
plan from `curvature/PLAN.md`, along with the pragmatic adjustments that were
needed to make the pipeline executable inside this repository.

## 1. Coverage Collection
- `curvature.coverage.CoverageCollector` instantiates Procgen environments
  via the same Hydra config used for training (one env per seed, always
  restarted at that seed).  
- For each seed it runs `steps_per_seed` environment steps, applies the PPO
  checkpoint policy, and mixes in the prescribed exploration noise:
  - ε-greedy with probability `epsilon` (default 0.20).
  - Sticky-action with probability `sticky_prob` (default 0.15) unless the
    previous step ended the episode.
- Observations are preprocessed with the loaded `Preprocessor` and passed
  through `SharedRepresentationExtractor`, which slices the shared IMPALA
  trunk right before the actor/critic head (Linear or Poincaré layer).  
- Every `subsample_stride` steps (default 4) we store the latent feature,
  timestamp, and seed id. Consecutive recorded states are linked to form the
  transition list used downstream. Transitions are reset whenever the env
  emits `first=1` to avoid cross-episode edges.

## 2. Latent Clustering & Graph Construction
- Features are z-scored (`whiten_features`) before clustering to match the
  “latent representation” definition in the plan.
- `MiniBatchKMeans` (default 10 k clusters, batch 8 192) produces cluster
  assignments and centroids; assignments map each recorded state to a node.
- `TransitionGraphBuilder` counts cluster-to-cluster transitions, drops
  self-loops, retains the top‑30 outgoing edges per node and row-normalizes
  to obtain transition probabilities. Edge lengths are computed via
  `-log(ε + P(u→v))` with ε=1e‑6.  
- The graph is symmetrized by keeping the minimum length found in either
  direction, and exported both as adjacency lists (for sampling) and as a
  `scipy.sparse.csr_matrix` for Dijkstra.

## 3. Target Metric & Pair Sampling
- `PairSampler` produces the 1 000 000 (configurable) training/validation
  pairs with the requested 50/30/20 mixture:
  1. **Direct edges (50 %)** – sampled uniformly from the sparsified graph,
     target distance = edge length.
  2. **Two-hop pairs (30 %)** – sample u→v→w chains and add the two edge
     lengths to approximate the shortest path.
  3. **Random far pairs (20 %)** – select up to 512 random sources, run
     multi-source Dijkstra on the symmetric graph, and draw targets from the
     reachable nodes (filtering `inf` distances).
- Pairs with non-finite targets are dropped before training.

## 4. Hyperbolic Fitting & Curvature Search
- `HyperbolicCurvatureEstimator` holds node embeddings `x_u` (Euclidean
  parameters projected inside radius `(1-τ)/√c`) and a learnable curvature
  parameter `c = softplus(θ) + c_min`. The loss matches the plan:
  \[(d_H^{(c)}(x_u,x_v) - α d_{target}(u,v))^2 + λ \max(0, c\|x_u\|^2 - ρ)^2\]
  with `λ=1e-3`, `ρ=0.9`, `τ=1e-3`.
- `α` (scale alignment) is recomputed in closed form at the start of every
  outer iteration by iterating over the training pairs without gradients.
- Training loop mirrors the warm-start / inner-outer schedule:
  - **Inner loop**: 10 embedding steps per outer iteration using Adam
    (`embedding_lr`, default 1e‑2) on shuffled batches of size 4 096.
  - **Outer loop**: up to 50 iterations; each iteration evaluates validation
    relative distortion, runs two curvature-only Adam steps (`curvature_lr`,
    default 2e‑3), and enforces the boundary projection.
  - **Monotonic guard**: three consecutive increases in validation relative
    distortion trigger a curvature LR decay by 0.5.
  - **Early stop**: if validation improvement stays below 0.25 % for `patience`
    (default 5) iterations, the loop halts.
- The best curvature/embedding checkpoint (lowest validation relative
  distortion) is recorded and returned alongside the full training history.

## 5. CLI Workflow (`curvature/estimate_curvature.py`)
1. `python curvature/estimate_curvature.py --env bigfish --checkpoint ...`
   loads the specified Hydra config + checkpoint.
2. Runs coverage, clustering, graph building, pair sampling, and the
   hyperbolic optimizer end-to-end.
3. Writes artifacts to `curvature_runs/<env>_<timestamp>/`, including (when
   `--save-intermediates` is set):
   - `coverage.npz`, `clustering.npz`, `pairs.npz`
   - `curvature_summary.json` (final `c*`, α, coverage stats)
   - `training_history.json`
   - `best_state.npz` (embeddings + θ snapshot)
4. Prints the optimal curvature `c*` for the requested environment.

## Deviations & Practical Considerations
- Riemannian Adam is approximated with Euclidean Adam + Poincaré projection
  because the curvature parameter is itself learnable (geoopt requires fixed
  curvature per manifold instance).
- Random pair sampling limits the number of Dijkstra solves by batching
  sources; this preserves coverage while keeping runtime feasible.
- Seeds are drawn randomly (without replacement) rather than sequentially to
  avoid aliasing with NumPy’s RNG state.
- Coverage transitions are reset across episode boundaries to prevent
  artificial long edges caused by automatic Procgen resets.

These adjustments keep the spirit of the original plan intact while ensuring
the code runs comfortably inside the existing project structure.
