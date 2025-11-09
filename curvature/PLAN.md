# Plan: Coverage Policy → Weighted State Graph → Best-Fit Poincaré Curvature (single-path strategy)

## 1) Main idea (no forks)
Estimate the curvature \(-c\) that best matches the environment’s policy-agnostic geometry by: (a) collecting broad coverage rollouts, (b) constructing a **clustered weighted transition graph** in latent space, (c) defining a **single** target metric from those transitions, and (d) **jointly optimizing** node embeddings **and** a **continuous** curvature parameter \(c>0\) to minimize scale-aligned distortion on held-out pairs.

---

## 2) Coverage policy (fixed design)
- **Base policy:** a mid-training PPO checkpoint (competent but not specialized).
- **Exploration noise:** with probability \(\epsilon=0.20\) replace the chosen action with a random action; with probability \(p_{\text{sticky}}=0.15\) repeat the previous action.
- **Seeds:** 500 Procgen level seeds.
- **Steps/seed:** 10,000 steps; **subsample** every 4 frames.
- **Recorded per step:** \((s_t, a_t, r_t, s_{t+1}, \text{seed}, t)\).

Rationale: ensures broad visitation while retaining coherent transitions.

---

## 3) Latent representation and clustering (fixed design)
- **Embedding \(\phi(s)\):** shared trunk activation immediately **before** the actor/critic heads; z-score over the entire coverage set.
- **Clustering:** k-means with \(k=10{,}000\).
- **Nodes:** cluster centroids \(\mu_u\) are nodes; each state \(s\) maps to node \(u=\arg\min\|\phi(s)-\mu_u\|_2\).

---

## 4) Weighted directed transition graph (fixed design)
- **Edge counts:** for each observed transition \(s_t\!\to\!s_{t+1}\), increment \(C(u\!\to\!v)\) using the assigned nodes \(u,v\).
- **Row-normalize:** \(P(u\!\to\!v)=C(u\!\to\!v)/\sum_{v'}C(u\!\to\!v')\).
- **Sparsify:** keep **top-30** outgoing edges per node by \(C(u\!\to\!v)\); drop self-loops.

---

## 5) Target metric (single choice)
Convert transition probabilities to **edge lengths** via **negative log-probability**:
\[
\ell(u\!\to\!v)=-\log(\epsilon + P(u\!\to\!v)),\quad \epsilon=10^{-6}.
\]
**Symmetrize** to undirected lengths:
\[
\ell_{\text{sym}}(u,v)=\min\big(\ell(u\!\to\!v),\,\ell(v\!\to\!u)\big),
\]
and compute **shortest-path distances** \(d_{\text{target}}(u,v)\) on the symmetrized graph.
- **Pair set:** sample 1,000,000 pairs with a 50/30/20 mix of (edges / two-hop / random far pairs).
- **Pair weights:** **uniform** (policy-agnostic geometry).

---

## 6) Poincaré embedding model (fixed design)
- **Space:** Poincaré ball \(\mathbb{B}^d_c\) with dimension \(d=16\).
- **Distance:**
\[
d_H^{(c)}(x,y)=\frac{1}{\sqrt{c}}\operatorname{arcosh}\!\left(1+\frac{2c\|x-y\|^2}{(1-c\|x\|^2)(1-c\|y\|^2)}\right).
\]
- **Variables:** node coordinates \(\{x_u\}\) and **curvature** \(c>0\).
- **Scale alignment:** learn a scalar \(\alpha>0\) to align units (closed-form least-squares each epoch).
- **Boundary safety:** project to radius \((1-\tau)/\sqrt{c}\) with \(\tau=10^{-3}\) after each update.

---

## 7) Intelligent curvature search (no grids)
**Jointly optimize \(c\) and embeddings** with a bilevel-style schedule:
- **Parameterization:** \(c=\mathrm{softplus}(\theta)+c_{\min}\), with \(c_{\min}=10^{-3}\).
- **Splits:** 90% of pairs for training, 10% for validation (fixed at start).
- **Loss (training pairs):**
\[
\mathcal{L}_{\text{train}}(\{x\},\theta)=
\frac{1}{|\mathcal{P}_{\text{tr}}|}\sum_{(u,v)\in\mathcal{P}_{\text{tr}}}\!\big(d_H^{(c)}(x_u,x_v)-\alpha\,d_{\text{target}}(u,v)\big)^2
+\lambda\,\sum_u \mathbf{1}\!\left[c\|x_u\|^2>\rho\right]\!(c\|x_u\|^2-\rho)^2,
\]
with \(\rho=0.9\), \(\lambda=10^{-3}\). Update \(\alpha\) in closed-form at the start of each epoch.

**Optimization loop (early-stopped):**
1. **Warm start:** initialize \(\theta\) so \(c\approx 1\); initialize \(x_u\) as small-norm Gaussians, then project.
2. **Inner updates (embeddings):** for 10 steps, optimize \(\{x\}\) with Riemannian Adam (batch 4096 pairs).
3. **Outer update (curvature):** freeze \(\{x\}\), take 1–2 Adam steps on \(\theta\) to reduce **validation** distortion
   \[
   \mathrm{RelDist}_{\text{val}}=\frac{\sum_{(u,v)\in\mathcal{P}_{\text{val}}}\!\big|d_H^{(c)}(x_u,x_v)-\alpha\,d_{\text{target}}(u,v)\big|}
   {\sum_{(u,v)\in\mathcal{P}_{\text{val}}} d_{\text{target}}(u,v)}.
   \]
   (Backprop through \(d_H^{(c)}\); recompute \(\alpha\) on **train** pairs each epoch.)
4. **Monotonicity guard:** if \(\mathrm{RelDist}_{\text{val}}\) increases for 3 consecutive outer steps, **reduce** curvature step size; if it still increases on the next check, **revert** the last \(\theta\) update (checkpoint), halve the step size, and continue.
5. **Convergence:** stop when \(\mathrm{RelDist}_{\text{val}}\) fails to improve by ≥0.25% over 5 outer steps or after 50 outer steps (whichever first).

**Outcome:** a **single** \(c^\*\) (decoded from \(\theta\)) and embeddings \(\{x_u\}\) that minimize validation distortion **without** any pre-selected curvature list.

---

## 8) Minimal hyperparameters (fixed)
- \(d=16\), batch size \(4096\), inner steps \(=10\), outer steps \(≤50\).
- Optimizer: Adam (embeddings lr \(1\mathrm{e}{-2}\), curvature lr \(2\mathrm{e}{-3}\)); cosine decay on embedding lr; curvature lr halves on monotonicity guard trigger.
- Projection margin \(\tau=1\mathrm{e}{-3}\); boundary \(\rho=0.9\); barrier \(\lambda=1\mathrm{e}{-3}\).

---

## 9) Interpretation
- The learned \(c^\*\) is a **continuous** best-fit curvature reflecting how well a hyperbolic model can represent shortest-path structure induced by coverage transitions.  
- Larger \(c^\*\) → stronger negative curvature → more tree-like expansion in the coverage-induced geometry.

