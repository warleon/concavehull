"""3D generalization of DTSCAN (Delaunay Triangulation-based Spatial
Clustering of Applications with Noise):

    Kim, J., & Cho, J. (2019). Delaunay Triangulation-Based Spatial
    Clustering Technique for Enhanced Adjacent Boundary Detection and
    Segmentation of LiDAR 3D Point Clouds. Sensors, 19(18), 3926.

`dtscan.hpp` (the 2D sibling of this module) implements the paper exactly
as published, over a 2D Delaunay triangulation: triangles pruned by a
z-score on their own AREA and on their longest edge's LENGTH, then DBSCAN's
region-growing over the surviving triangulation-adjacency graph.

The paper itself targets 3D LiDAR point clouds, but its own method section
describes the 2D case (triangles/edges) throughout. This module is the
direct 3D generalization of that same two-stage method: triangles become
tetrahedra, and "area" becomes "volume" -- the natural higher-dimensional
analogue, computed over a true 3D Delaunay triangulation -- while "edge
length" is unchanged (an edge is still a 1-simplex regardless of ambient
dimension). Stage 2 (DBSCAN region growing over triangulation-adjacency)
is dimension-agnostic and carries over unmodified.

A 3D Delaunay triangulation has no equivalent of `delaunator` (2D-only,
used by `dtscan.hpp`) in this project, so this module builds it via GUDHI's
`AlphaComplex`, which is CGAL's Delaunay triangulation under the hood: with
`max_alpha_square=inf`, every simplex belongs to the final complex, i.e.
the complete (unfiltered) Delaunay triangulation, not an alpha-shape
subset.
"""

from __future__ import annotations

from dataclasses import dataclass

import gudhi
import numpy as np


@dataclass
class DTScan3DResult:
    # labels[i] is the 0-based cluster id of points[i], or -1 if noise.
    labels: np.ndarray
    n_clusters: int
    # The input points, and the surviving mesh (tetrahedra that passed
    # stage-1 z-score pruning) as an (T, 4) array of vertex indices into
    # `points` -- the 3D analogue of the 2D `Graph`/`compute_graph()`
    # output: the actual kept triangulation, not just cluster labels.
    points: np.ndarray
    tetrahedra: np.ndarray


def _tetrahedron_volume(p0, p1, p2, p3):
    return abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


def dtscan_3d(points, volume_z_threshold: float = 1.0, length_z_threshold: float = 1.0,
              min_pts: int = 3) -> DTScan3DResult:
    """3D DTSCAN: z-score-prune a 3D Delaunay triangulation's tetrahedra by
    volume/edge-length (Eqs. 2-3 of the paper, generalized from triangles
    /area), then run DBSCAN's region-growing over the surviving
    triangulation-adjacency graph. Mirrors `dtscan()` (2D) in both
    signature and return shape -- see `tests/test_dtscan_3d.py` for the
    same battery of correctness properties.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    labels = np.full(n, -1, dtype=int)
    empty_tets = np.empty((0, 4), dtype=int)
    if n < 4:
        return DTScan3DResult(labels=labels, n_clusters=0, points=points, tetrahedra=empty_tets)

    alpha_complex = gudhi.AlphaComplex(points=points)
    simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=float("inf"))

    tetrahedra = [s for s, _ in simplex_tree.get_simplices() if len(s) == 4]
    if not tetrahedra:
        return DTScan3DResult(labels=labels, n_clusters=0, points=points, tetrahedra=empty_tets)

    # --- Stage 1: z-score pruning (volume generalizes "area"; edge length
    # is unchanged) ---
    volumes = np.array([_tetrahedron_volume(*[points[v] for v in tet]) for tet in tetrahedra])
    volume_mean, volume_std = volumes.mean(), volumes.std()

    edge_length = {}
    for s, _ in simplex_tree.get_skeleton(1):
        if len(s) == 2:
            a, b = s
            edge_length[(a, b)] = float(np.linalg.norm(points[a] - points[b]))
    length_mean = np.mean(list(edge_length.values()))
    length_std = np.std(list(edge_length.values()))

    def tet_edges(tet):
        return [(tet[i], tet[j]) if tet[i] < tet[j] else (tet[j], tet[i])
                for i in range(4) for j in range(i + 1, 4)]

    survives = np.ones(len(tetrahedra), dtype=bool)
    for t, tet in enumerate(tetrahedra):
        volume_z = (volumes[t] - volume_mean) / volume_std if volume_std > 0 else 0.0
        max_len_z = -np.inf
        for e in tet_edges(tet):
            length = edge_length[e]
            length_z = (length - length_mean) / length_std if length_std > 0 else 0.0
            max_len_z = max(max_len_z, length_z)
        if volume_z > volume_z_threshold or max_len_z > length_z_threshold:
            survives[t] = False

    # Pruned adjacency: edges of every surviving tetrahedron only.
    neighbors = [set() for _ in range(n)]
    for t, tet in enumerate(tetrahedra):
        if not survives[t]:
            continue
        for a, b in tet_edges(tet):
            neighbors[a].add(b)
            neighbors[b].add(a)

    # --- Stage 2: DBSCAN region growing over triangulation-adjacency,
    # identical to dtscan.hpp's own stage 2. ---
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighbors[i]) < min_pts:
            continue  # not a core point -- tentatively noise

        labels[i] = cluster_id
        seeds = list(neighbors[i])
        idx = 0
        while idx < len(seeds):
            q = seeds[idx]
            idx += 1
            if not visited[q]:
                visited[q] = True
                if len(neighbors[q]) >= min_pts:
                    seeds.extend(neighbors[q])
            if labels[q] == -1:
                labels[q] = cluster_id
        cluster_id += 1

    surviving_tetrahedra = np.array([tet for t, tet in enumerate(tetrahedra) if survives[t]], dtype=int)
    if surviving_tetrahedra.size == 0:
        surviving_tetrahedra = empty_tets

    return DTScan3DResult(labels=labels, n_clusters=cluster_id, points=points, tetrahedra=surviving_tetrahedra)
