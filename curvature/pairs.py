from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.sparse.csgraph import dijkstra

from curvature.graph import GraphData


@dataclass
class PairDataset:
    pairs: np.ndarray  # [N, 2]
    distances: np.ndarray  # [N]
    weights: np.ndarray  # [N]
    categories: np.ndarray  # [N] encoded as ints: 0=edge,1=two-hop,2=random


class PairSampler:
    def __init__(self, graph: GraphData, rng_seed: int = 0):
        self.graph = graph
        self.rng = np.random.default_rng(rng_seed)
        self._length_lookup = [
            {int(neighbor): float(length) for neighbor, length in zip(neigh_list, length_list)}
            for neigh_list, length_list in zip(graph.neighbors, graph.lengths)
        ]

    def sample_pairs(
        self,
        total_pairs: int,
        edge_fraction: float = 0.5,
        two_hop_fraction: float = 0.3,
    ) -> PairDataset:
        n_edge = int(total_pairs * edge_fraction)
        n_two_hop = int(total_pairs * two_hop_fraction)
        n_random = max(0, total_pairs - n_edge - n_two_hop)

        edge_pairs, edge_dist = self._sample_direct_edges(n_edge)
        two_pairs, two_dist = self._sample_two_hop_pairs(n_two_hop)
        random_pairs, random_dist = self._sample_random_pairs(n_random)

        pairs = np.concatenate([edge_pairs, two_pairs, random_pairs], axis=0)
        distances = np.concatenate([edge_dist, two_dist, random_dist], axis=0)
        weights = np.ones_like(distances, dtype=np.float32)
        categories = np.concatenate(
            [
                np.zeros(edge_pairs.shape[0], dtype=np.int8),
                np.ones(two_pairs.shape[0], dtype=np.int8),
                np.full(random_pairs.shape[0], 2, dtype=np.int8),
            ],
            axis=0,
        )
        return PairDataset(pairs=pairs, distances=distances, weights=weights, categories=categories)

    def _sample_direct_edges(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        edges = self.graph.edge_index
        lengths = self.graph.edge_lengths
        if edges.size == 0 or n_samples == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        idx = self.rng.integers(low=0, high=len(edges), size=n_samples)
        return edges[idx], lengths[idx]

    def _sample_two_hop_pairs(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        if n_samples == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        pairs: List[Tuple[int, int]] = []
        lengths: List[float] = []
        attempts = 0
        max_attempts = max(1000, n_samples * 10)
        while len(pairs) < n_samples and attempts < max_attempts:
            attempts += 1
            u = int(self.rng.integers(low=0, high=self.graph.num_nodes))
            neighbors_u = self.graph.neighbors[u]
            if len(neighbors_u) == 0:
                continue
            v = int(neighbors_u[self.rng.integers(low=0, high=len(neighbors_u))])
            neighbors_v = self.graph.neighbors[v]
            if len(neighbors_v) == 0:
                continue
            w = int(neighbors_v[self.rng.integers(low=0, high=len(neighbors_v))])
            if u == w:
                continue
            len_uv = self._length_lookup[u].get(v)
            len_vw = self._length_lookup[v].get(w)
            if len_uv is None or len_vw is None:
                continue
            pairs.append((u, w))
            lengths.append(len_uv + len_vw)
        if len(pairs) == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        return np.array(pairs, dtype=np.int64), np.array(lengths, dtype=np.float32)

    def _sample_random_pairs(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        if n_samples == 0 or self.graph.csr.nnz == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)

        max_sources = min(512, self.graph.num_nodes)
        per_source = max(1, n_samples // max_sources)
        n_sources = min(max_sources, max(1, n_samples // per_source))
        sources = self.rng.choice(self.graph.num_nodes, size=n_sources, replace=False)
        distances_matrix = dijkstra(self.graph.csr, directed=False, indices=sources)
        pairs: List[Tuple[int, int]] = []
        distances: List[float] = []
        for src_idx, src in enumerate(sources):
            finite_mask = np.isfinite(distances_matrix[src_idx])
            finite_mask[src] = False
            reachable = np.where(finite_mask)[0]
            if reachable.size == 0:
                continue
            chosen = self.rng.choice(reachable, size=min(per_source, reachable.size), replace=False)
            for dst in chosen:
                pairs.append((int(src), int(dst)))
                distances.append(float(distances_matrix[src_idx, dst]))
                if len(pairs) >= n_samples:
                    break
            if len(pairs) >= n_samples:
                break
        if len(pairs) == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        return np.array(pairs, dtype=np.int64), np.array(distances, dtype=np.float32)
