from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class GraphData:
    num_nodes: int
    neighbors: List[np.ndarray]
    lengths: List[np.ndarray]
    probabilities: List[np.ndarray]
    counts: List[np.ndarray]
    csr: csr_matrix
    edge_index: np.ndarray
    edge_lengths: np.ndarray


class TransitionGraphBuilder:
    """
    Builds the clustered transition graph with sparsification + symmetrization.
    """

    def __init__(
        self,
        assignments: np.ndarray,
        transitions: np.ndarray,
        num_nodes: int,
        top_k: int = 30,
        epsilon: float = 1e-6,
    ):
        self.assignments = assignments
        self.transitions = transitions
        self.num_nodes = num_nodes
        self.top_k = top_k
        self.epsilon = epsilon

    def _count_edges(self) -> Dict[Tuple[int, int], int]:
        counts: Dict[Tuple[int, int], int] = defaultdict(int)
        for src_idx, dst_idx in self.transitions:
            u = int(self.assignments[src_idx])
            v = int(self.assignments[dst_idx])
            if u == v:
                continue
            counts[(u, v)] += 1
        return counts

    def build(self) -> GraphData:
        counts = self._count_edges()
        outgoing: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for (u, v), c in counts.items():
            outgoing[u].append((v, c))

        neighbors: List[np.ndarray] = []
        probs: List[np.ndarray] = []
        lengths: List[np.ndarray] = []
        kept_counts: List[np.ndarray] = []
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        # First pass: sparsify and compute directional lengths.
        edge_index = []
        edge_lengths = []

        for node in range(self.num_nodes):
            node_edges = outgoing.get(node, [])
            if node_edges:
                node_edges.sort(key=lambda item: item[1], reverse=True)
                node_edges = node_edges[: self.top_k]
                total = sum(c for _, c in node_edges)
                node_neighbors = np.array([v for v, _ in node_edges], dtype=np.int64)
                node_counts = np.array([c for _, c in node_edges], dtype=np.float32)
                node_probs = node_counts / (total + 1e-8)
                node_lengths = -np.log(self.epsilon + node_probs)
                neighbors.append(node_neighbors)
                probs.append(node_probs)
                lengths.append(node_lengths)
                kept_counts.append(node_counts)
                for v, length in zip(node_neighbors, node_lengths):
                    rows.append(node)
                    cols.append(int(v))
                    data.append(float(length))
                    edge_index.append((node, int(v)))
                    edge_lengths.append(float(length))
            else:
                neighbors.append(np.zeros((0,), dtype=np.int64))
                probs.append(np.zeros((0,), dtype=np.float32))
                lengths.append(np.zeros((0,), dtype=np.float32))
                kept_counts.append(np.zeros((0,), dtype=np.float32))

        directed = csr_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_nodes))

        # Symmetrize by taking the min directional cost for each undirected edge.
        sym_rows: List[int] = []
        sym_cols: List[int] = []
        sym_data: List[float] = []
        visited = set()
        for (u, v), length_uv in zip(edge_index, edge_lengths):
            if (u, v) in visited:
                continue
            reverse_val = directed[v, u]
            if reverse_val != 0:
                if isinstance(reverse_val, np.matrix) or hasattr(reverse_val, "A"):
                    reverse_scalar = float(reverse_val.A[0][0])
                else:
                    reverse_scalar = float(reverse_val)
                merged = min(length_uv, reverse_scalar)
            else:
                merged = length_uv
            sym_rows.extend([u, v])
            sym_cols.extend([v, u])
            sym_data.extend([merged, merged])
            visited.add((u, v))
            visited.add((v, u))

        csr_sym = csr_matrix((sym_data, (sym_rows, sym_cols)), shape=(self.num_nodes, self.num_nodes))
        return GraphData(
            num_nodes=self.num_nodes,
            neighbors=neighbors,
            lengths=lengths,
            probabilities=probs,
            counts=kept_counts,
            csr=csr_sym,
            edge_index=np.array(edge_index, dtype=np.int64) if edge_index else np.zeros((0, 2), dtype=np.int64),
            edge_lengths=np.array(edge_lengths, dtype=np.float32) if edge_lengths else np.zeros((0,), dtype=np.float32),
        )
