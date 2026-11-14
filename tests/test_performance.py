"""Performance regression tests for ChiOutline.

Not strict timing assertions (too flaky across machines/CI) -- instead
assert that each input completes within a generous wall-clock budget. Both
scenarios were benchmarked directly (not assumed) against the
pre-optimization code; these guard against either class of regression
regressing silently.
"""

import time

import numpy as np

from concavehull import ChiOutline


def test_dense_single_cluster_completes_within_budget():
    # Adversarial: one dense cluster, low chi (maximizes isolating-edge
    # removals). Before the O(1) alive_triangle_count + count_boundary_cycles
    # optimizations, this class of input was measurably ~2.2-2.4x slower.
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((2000, 2))

    start = time.time()
    co = ChiOutline(pts, chi=0.2)
    polys = co.compute()
    elapsed = time.time() - start

    assert len(polys) >= 1
    # Measured ~0.13s post-optimization for this exact input on a typical
    # dev machine; a generous 5s budget catches a severe (not marginal)
    # regression without being flaky on slower/loaded CI runners.
    assert elapsed < 5.0, f"ChiOutline took {elapsed:.2f}s on 2000 points (chi=0.2); expected well under 5s"


def test_many_small_clusters_completes_within_budget():
    # Realistic: many separate, small clusters scattered over a large area
    # (object-detection-bounding-box-midpoint-like), rather than one single
    # blob. Before per-component indexing (component_triangles/
    # component_boundary_edges, running sigma_bar, and scoped boundary-key
    # rebuilding), find_pieces()/count_boundary_cycles() filtered a
    # component's own members out of the GLOBAL triangle/boundary
    # collections on every call -- so despite only running on confirmed
    # splits, their cost scaled with the combined size of every
    # simultaneously-live component, not just the one actually being
    # queried. With many small clusters (common in real detection data),
    # this showed clearly super-linear growth; measured ~2-2.7x faster
    # after the fix, with growth noticeably closer to O(n log n).
    rng = np.random.default_rng(7)
    n_clusters = 2000
    pts_per_cluster = 8
    centers = rng.uniform(low=0, high=2000, size=(n_clusters, 2))
    pts = np.vstack(
        [rng.normal(loc=c, scale=5.0, size=(pts_per_cluster, 2)) for c in centers]
    )

    start = time.time()
    co = ChiOutline(pts, chi=1.0)
    polys = co.compute()
    elapsed = time.time() - start

    assert len(polys) >= 1
    # Measured ~0.6s post-optimization for this exact input (16000 points,
    # 2000 clusters) on a typical dev machine; a generous 10s budget catches
    # a severe regression without being flaky on slower/loaded CI runners.
    assert elapsed < 10.0, (
        f"ChiOutline took {elapsed:.2f}s on {len(pts)} points across {n_clusters} clusters; "
        "expected well under 10s"
    )
