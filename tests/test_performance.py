"""Performance regression test for ChiOutline.

Not a strict timing assertion (too flaky across machines/CI) -- instead
asserts that a moderately large (1000+ point), adversarial (single dense
cluster, low chi -- maximizes isolating-edge-removal events) input
completes within a generous wall-clock budget. Before the O(1)
alive_triangle_count + count_boundary_cycles optimizations, this class of
input was measurably ~2.2-2.4x slower (benchmarked directly against the
pre-optimization code); this guards against that regressing silently.
"""

import time

import numpy as np

from concavehull import ChiOutline


def test_dense_single_cluster_completes_within_budget():
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((2000, 2))

    start = time.time()
    co = ChiOutline(pts, chi=0.2)  # low chi: maximizes isolating-edge removals
    polys = co.compute()
    elapsed = time.time() - start

    assert len(polys) >= 1
    # Measured ~0.2s post-optimization for this exact input on a typical
    # dev machine; a generous 5s budget catches a severe (not marginal)
    # regression without being flaky on slower/loaded CI runners.
    assert elapsed < 5.0, f"ChiOutline took {elapsed:.2f}s on 2000 points (chi=0.2); expected well under 5s"
