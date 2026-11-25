"""Correctness tests for dtscan_3d() -- the 3D generalization of DTSCAN:

    Kim, J., & Cho, J. (2019). Delaunay Triangulation-Based Spatial
    Clustering Technique for Enhanced Adjacent Boundary Detection and
    Segmentation of LiDAR 3D Point Clouds. Sensors, 19(18), 3926.

dtscan.hpp (2D) implements the paper exactly as published: triangles
pruned by a z-score on their own AREA and longest EDGE LENGTH, then
DBSCAN's region-growing over triangulation-adjacency. This module is the
direct 3D analogue -- triangles become tetrahedra from a true 3D Delaunay
triangulation (via GUDHI), "area" generalizes to "volume", and edge length
/region growing are otherwise unchanged. See concavehull/three_d/dtscan_3d.py
for the implementation note on why GUDHI (not the project's own 2D-only
delaunator) is used here."""

import numpy as np

from concavehull.three_d import dtscan_3d


def test_four_separated_blobs_form_four_clean_clusters():
    rng = np.random.default_rng(1)
    centers = [(0, 0, 0), (15, 0, 0), (7.5, 13, 0), (7.5, 4, 13)]
    pts = np.vstack([rng.normal(c, 0.5, size=(60, 3)) for c in centers])

    r = dtscan_3d(pts, volume_z_threshold=1.0, length_z_threshold=1.0, min_pts=4)

    assert r.labels.shape == (240,)
    assert r.n_clusters == 4
    distinct = set(r.labels.tolist()) - {-1}
    assert len(distinct) == 4
    for i in range(4):
        blob_labels = set(r.labels[i * 60 : (i + 1) * 60].tolist()) - {-1}
        assert len(blob_labels) == 1


def test_resists_the_chaining_problem():
    # Same "touching problem" property test_dtscan.py's own
    # test_resists_the_chaining_problem checks in 2D: two dense blobs
    # joined by a thin bridge of sparse points should not be merged by
    # the z-score pruning + triangulation-adjacency neighbor rule.
    rng = np.random.default_rng(2)
    pts_a = rng.normal([0, 0, 0], 0.4, size=(60, 3))
    pts_b = rng.normal([12, 0, 0], 0.4, size=(60, 3))
    bridge = np.array([[12 * i / 6, 0.0, 0.0] for i in range(1, 6)])
    pts = np.vstack([pts_a, pts_b, bridge])

    r = dtscan_3d(pts, volume_z_threshold=1.0, length_z_threshold=1.0, min_pts=4)

    label_a, label_b = r.labels[0], r.labels[60]
    assert label_a != -1
    assert label_b != -1
    assert label_a != label_b


def test_too_few_points_is_all_noise_no_crash():
    # A 3D Delaunay triangulation needs >= 4 points (a single tetrahedron);
    # dtscan_3d's own guard returns all-noise rather than calling into
    # GUDHI with too few points.
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    r = dtscan_3d(pts)
    assert r.labels.shape == (3,)
    assert np.all(r.labels == -1)
    assert r.n_clusters == 0


def test_labels_are_dense_ids_from_zero():
    rng = np.random.default_rng(1)
    centers = [(0, 0, 0), (15, 0, 0), (7.5, 13, 0), (7.5, 4, 13)]
    pts = np.vstack([rng.normal(c, 0.5, size=(60, 3)) for c in centers])
    r = dtscan_3d(pts, volume_z_threshold=1.0, length_z_threshold=1.0, min_pts=4)
    distinct = sorted(set(r.labels.tolist()) - {-1})
    assert distinct == list(range(len(distinct)))
    assert r.n_clusters == len(distinct)


def test_aggressive_negative_threshold_prunes_most_of_uniform_scatter_to_noise():
    # Mirrors test_dtscan.py's own sparse-scatter property: a meaningfully
    # negative z-score threshold (well below average) is needed to force
    # most of a roughly-homogeneous random layout to noise, since a
    # Delaunay triangulation gives every point many neighbors regardless
    # of how "random" the layout is.
    rng = np.random.default_rng(3)
    pts = rng.uniform(-50, 50, size=(40, 3))
    r = dtscan_3d(pts, volume_z_threshold=-0.5, length_z_threshold=-0.5, min_pts=4)
    assert (r.labels == -1).sum() >= 25


def test_a_single_tight_cluster_is_one_cluster_not_shattered():
    rng = np.random.default_rng(4)
    pts = rng.normal([0, 0, 0], 0.5, size=(100, 3))
    r = dtscan_3d(pts, volume_z_threshold=1.5, length_z_threshold=1.5, min_pts=4)
    distinct = set(r.labels.tolist()) - {-1}
    assert len(distinct) == 1
    assert (r.labels != -1).sum() > 80


def test_result_exposes_points_and_surviving_tetrahedra():
    # The 3D analogue of the 2D Graph/compute_graph() output: the actual
    # kept mesh (tetrahedra surviving stage-1 z-score pruning), not just
    # cluster labels.
    rng = np.random.default_rng(1)
    centers = [(0, 0, 0), (15, 0, 0), (7.5, 13, 0), (7.5, 4, 13)]
    pts = np.vstack([rng.normal(c, 0.5, size=(60, 3)) for c in centers])

    r = dtscan_3d(pts, volume_z_threshold=1.0, length_z_threshold=1.0, min_pts=4)

    assert np.array_equal(r.points, pts)
    assert r.tetrahedra.ndim == 2 and r.tetrahedra.shape[1] == 4
    assert len(r.tetrahedra) > 0
    assert r.tetrahedra.min() >= 0
    assert r.tetrahedra.max() < len(pts)

    # a surviving tetrahedron is, by construction, a tight local
    # neighborhood -- it should never straddle two DIFFERENT real clusters
    # (it may still touch noise-labeled vertices, e.g. right at a cluster's
    # own fringe).
    for tet in r.tetrahedra:
        real_labels = set(r.labels[tet].tolist()) - {-1}
        assert len(real_labels) <= 1


def test_too_few_points_returns_empty_mesh():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    r = dtscan_3d(pts)
    assert r.points.shape == (3, 3)
    assert r.tetrahedra.shape == (0, 4)
