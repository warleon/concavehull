"""Correctness tests for ChiOutline (Zhong & Duckham, 2016).

Several of these are regression tests for specific bugs found during
development, not just "does it run" smoke tests:

  - negative chi must not raise (chi is an unbounded mu+chi*sigma offset,
    not a normalized [0,1] fraction -- see chioutline.hpp's own comment).
  - extract_polygons() once filtered an output polygon by the combined
    vertex count of every edge sharing its component id, rather than each
    walked boundary cycle's own count, letting a tiny spurious fragment
    ride through the size filter alongside a large legitimate polygon.
  - the O(1) alive_triangle_count bookkeeping and the count_boundary_cycles
    pre-check (added for performance) must not change any of the above.
"""

import numpy as np
import pytest

from concavehull import ChiOutline


def two_blobs(seed=0, n_per_blob=200, separation=15.0, scale=1.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[0, 0], scale=scale, size=(n_per_blob, 2))
    b = rng.normal(loc=[separation, 0], scale=scale, size=(n_per_blob, 2))
    return np.vstack([a, b])


def three_blobs_with_outliers(seed=42):
    rng = np.random.default_rng(seed)
    centers = [(0, 0), (10, 0), (5, 10)]
    clusters = [rng.normal(loc=c, scale=1.5, size=(90, 2)) for c in centers]
    outliers = rng.uniform(low=-5, high=20, size=(40, 2))
    return np.vstack(clusters + [outliers])


def test_negative_chi_does_not_raise():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((300, 2))
    co = ChiOutline(pts, chi=-0.172)
    polys = co.compute()  # must not raise
    assert isinstance(polys, list)


def test_single_blob_stays_one_polygon_at_moderate_chi():
    rng = np.random.default_rng(11)
    pts = rng.standard_normal((300, 2))
    co = ChiOutline(pts, chi=0.6)
    polys = co.compute()
    assert len(polys) == 1


def test_three_blobs_with_outliers_split_correctly():
    pts = three_blobs_with_outliers()
    co = ChiOutline(pts, chi=1.3)
    polys = co.compute()
    assert len(polys) == 3


def min_component_points(n_total_points: int) -> int:
    """Mirrors ChiOutline's own min_component_points(): round(ln(n)), matching
    Eq. 2's small-component filter."""
    return max(0, round(np.log(n_total_points)))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_no_polygon_ever_violates_the_small_component_size_filter(seed):
    # Regression test for a real bug: extract_polygons() used to filter each
    # output polygon by the *combined* vertex count of every edge sharing
    # its component id, rather than each walked boundary cycle's own count.
    # Concretely, at seed=0, chi=0.4345 this produced a 3-vertex fragment
    # riding alongside two legitimate polygons (sizes [45, 50, 3]) -- 3 is
    # below min_component_points() for n=400 (6), so it should have been
    # filtered out and wasn't. Sweep several chi values and seeds to guard
    # against the same class of bug reappearing at a different combination.
    # NOTE: this does NOT assert "exactly 2 polygons" -- at low chi,
    # chi-shape/chi-outline legitimately carve many small (but >= the size
    # filter) fragments from ordinary boundary noise, even for a clean
    # two-blob input; that's documented, expected behavior, not a bug.
    pts = two_blobs(seed=seed)
    expected_min = min_component_points(len(pts))
    for chi in [0.2, 0.3, 0.4345, 0.5, 0.6, 0.8, 1.0]:
        co = ChiOutline(pts, chi=chi)
        polys = co.compute()
        for poly in polys:
            n_verts = len(poly)
            assert n_verts >= expected_min, (
                f"seed={seed} chi={chi}: polygon with {n_verts} vertices "
                f"violates the size filter (min={expected_min})"
            )


def test_known_bug_repro_now_gives_exactly_two_clean_polygons():
    # The exact input/chi combination that exposed the bug above: verify it
    # now gives the correct result (2 polygons), not just "no filter
    # violation" -- a tighter check on this one specific known-bad case.
    pts = two_blobs(seed=0, n_per_blob=200, separation=15.0)
    co = ChiOutline(pts, chi=0.4345)
    polys = co.compute()
    assert len(polys) == 2


def test_every_returned_polygon_has_at_least_3_vertices():
    for pts in [two_blobs(seed=s) for s in range(4)] + [three_blobs_with_outliers()]:
        for chi in [0.3, 0.6, 1.0, 1.3]:
            co = ChiOutline(pts, chi=chi)
            for poly in co.compute():
                assert len(poly) >= 3


def test_outliers_are_excluded_from_every_polygon():
    rng = np.random.default_rng(5)
    cluster = rng.normal(loc=[0, 0], scale=0.3, size=(100, 2))
    outliers = rng.uniform(low=-5, high=5, size=(15, 2))
    pts = np.vstack([cluster, outliers])

    co = ChiOutline(pts, chi=0.6)
    polys = co.compute()
    assert len(polys) >= 1

    all_vertices = np.vstack(polys) if polys else np.empty((0, 2))
    far_outliers = outliers[np.linalg.norm(outliers, axis=1) > 2.0]
    for outlier in far_outliers:
        assert not np.any(np.all(np.isclose(all_vertices, outlier), axis=1))


class TestChiLengthConversion:
    def test_round_trip(self):
        pts = three_blobs_with_outliers()
        length = ChiOutline.chi_to_length(pts, 1.3)
        chi_back = ChiOutline.length_to_chi(pts, length)
        assert chi_back == pytest.approx(1.3, abs=1e-9)

    def test_length_to_chi_can_be_negative(self):
        # Regression context: a length below the point set's mean DT edge
        # length legitimately produces a negative chi; this must round-trip
        # cleanly and must not raise.
        pts = two_blobs(seed=0)
        chi = ChiOutline.length_to_chi(pts, length=0.01)
        assert chi < 0
        ChiOutline(pts, chi=chi).compute()  # must not raise

    def test_max_meaningful_chi_saturates_to_convex_hull(self):
        pts = three_blobs_with_outliers()
        max_chi = ChiOutline.max_meaningful_chi(pts)
        co = ChiOutline(pts, chi=max_chi + 1.0)
        polys = co.compute()
        assert len(polys) == 1  # at/beyond saturation: one connected (convex-hull-like) region


class TestElbowChi:
    def test_single_blob_does_not_split(self):
        rng = np.random.default_rng(1)
        pts = rng.standard_normal((300, 2))
        chi = ChiOutline.elbow_chi(pts)
        polys = ChiOutline(pts, chi=chi).compute()
        assert len(polys) == 1

    def test_two_separated_blobs_split_into_two(self):
        pts = two_blobs(seed=0, n_per_blob=200, separation=15.0)
        chi = ChiOutline.elbow_chi(pts)
        polys = ChiOutline(pts, chi=chi).compute()
        assert len(polys) == 2

    def test_three_blobs_with_outliers_in_reasonable_range(self):
        pts = three_blobs_with_outliers()
        chi = ChiOutline.elbow_chi(pts)
        polys = ChiOutline(pts, chi=chi).compute()
        assert 2 <= len(polys) <= 4

    def test_does_not_raise_on_sparse_scattered_points(self):
        # The scenario that broke the old bbox-diagonal-based default: a
        # handful of sparse points over a large area (e.g. a few object
        # detections scattered across a photo). elbow_chi() must still
        # return a finite, usable value.
        rng = np.random.default_rng(3)
        pts = rng.uniform(low=0, high=800, size=(5, 2))
        chi = ChiOutline.elbow_chi(pts)
        assert np.isfinite(chi)
        ChiOutline(pts, chi=chi).compute()  # must not raise

    def test_scale_invariant(self):
        # elbow_chi is derived purely from the point set's own edge-length
        # distribution (ranks + normalized lengths), so scaling every
        # coordinate by a constant factor must not change the result.
        pts = two_blobs(seed=0)
        chi_a = ChiOutline.elbow_chi(pts)
        chi_b = ChiOutline.elbow_chi(pts * 37.5)
        assert chi_a == pytest.approx(chi_b, rel=1e-6)
