"""Correctness tests for dtscan() -- a faithful implementation of DTSCAN:

    Kim, J., & Cho, J. (2019). Delaunay Triangulation-Based Spatial
    Clustering Technique for Enhanced Adjacent Boundary Detection and
    Segmentation of LiDAR 3D Point Clouds. Sensors, 19(18), 3926.

Transcribed from the paper's own PDF (Section 3), not reconstructed from
secondary summaries: (1) z-score-prune Delaunay triangles by area/edge
length, (2) run DBSCAN's region-growing over the survivors using
triangulation-adjacency as the neighbor relation."""

import numpy as np

from concavehull import dtscan


def test_three_separated_blobs_form_three_clean_clusters():
    rng = np.random.default_rng(1)
    centers = [(0, 0), (15, 0), (7.5, 13)]
    pts = np.vstack([rng.normal([cx, cy], 0.5, size=(80, 2)) for cx, cy in centers])

    labels = dtscan(pts, area_z_threshold=1.0, length_z_threshold=1.0, min_pts=4)

    assert labels.shape == (240,)
    distinct = set(labels.tolist()) - {-1}
    assert len(distinct) == 3
    # every blob's own points should be assigned a single, consistent label
    for i in range(3):
        blob_labels = set(labels[i * 80 : (i + 1) * 80].tolist()) - {-1}
        assert len(blob_labels) == 1


def test_resists_the_chaining_problem():
    # Two dense blobs joined by a thin chain of sparse bridging points -- the
    # exact "touching problem" (Figure 2 of the paper) DTSCAN's z-score
    # pruning + triangulation-adjacency neighbor rule is designed to survive,
    # unlike a naive single-linkage/distance-threshold clustering, which
    # would merge the two blobs via the chain.
    rng = np.random.default_rng(2)
    pts_a = rng.normal([0, 0], 0.4, size=(60, 2))
    pts_b = rng.normal([12, 0], 0.4, size=(60, 2))
    bridge = np.array([[12 * i / 6, 0.0] for i in range(1, 6)])
    pts = np.vstack([pts_a, pts_b, bridge])

    labels = dtscan(pts, area_z_threshold=1.0, length_z_threshold=1.0, min_pts=4)

    label_a, label_b = labels[0], labels[60]
    assert label_a != -1
    assert label_b != -1
    assert label_a != label_b


def test_sparse_scattered_points_are_mostly_noise_at_an_aggressive_threshold():
    # Verified directly (not assumed): at the default-ish threshold of 1.0,
    # uniformly scattered points do NOT reliably become noise here -- a
    # Delaunay triangulation gives every point ~6 neighbors on average
    # regardless of how "random" the layout is, and z-score pruning relative
    # to the point set's OWN global mean/std doesn't aggressively prune edges
    # for a roughly homogeneous (no real multi-density structure) layout, so
    # most points still satisfy a loose min_pts=4 and end up in one cluster
    # -- the same behavior plain DBSCAN shows on uniform data with a loose
    # eps/MinPts. A meaningfully negative threshold (an aggressive,
    # below-average cut) is needed to force most of this scattered layout to
    # noise.
    rng = np.random.default_rng(3)
    pts = rng.uniform(-50, 50, size=(30, 2))
    labels = dtscan(pts, area_z_threshold=-0.5, length_z_threshold=-0.5, min_pts=4)
    assert (labels == -1).sum() >= 20


def test_too_few_points_is_all_noise_no_crash():
    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    labels = dtscan(pts)
    assert labels.shape == (2,)
    assert np.all(labels == -1)


def test_labels_are_dense_ids_from_zero():
    rng = np.random.default_rng(1)
    centers = [(0, 0), (15, 0), (7.5, 13)]
    pts = np.vstack([rng.normal([cx, cy], 0.5, size=(80, 2)) for cx, cy in centers])
    labels = dtscan(pts, area_z_threshold=1.0, length_z_threshold=1.0, min_pts=4)
    distinct = sorted(set(labels.tolist()) - {-1})
    assert distinct == list(range(len(distinct)))
