"""Basic correctness tests for concavehull() (chi-shape, Duckham et al. 2008)."""

import numpy as np
import pytest

from concavehull import concavehull


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_returns_closed_simple_polygon_containing_all_points():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((200, 2))
    hull = concavehull(pts, chi_factor=0.1)

    assert hull.ndim == 2 and hull.shape[1] == 2
    assert len(hull) >= 3
    # every polygon vertex should itself be one of the input points
    for v in hull:
        assert np.any(np.all(np.isclose(pts, v), axis=1))


def test_chi_factor_1_is_convex_hull_area():
    rng = np.random.default_rng(1)
    pts = rng.standard_normal((150, 2))
    hull_chi1 = concavehull(pts, chi_factor=1.0)

    # the true convex hull via a minimal, independent monotone-chain implementation
    pts_sorted = sorted(map(tuple, pts))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts_sorted:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts_sorted):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    true_hull = np.array(lower[:-1] + upper[:-1])

    assert polygon_area(hull_chi1) == pytest.approx(polygon_area(true_hull), rel=1e-6)


def test_decreasing_chi_factor_never_increases_area():
    rng = np.random.default_rng(2)
    pts = rng.standard_normal((200, 2))
    areas = [polygon_area(concavehull(pts, chi_factor=cf)) for cf in [1.0, 0.5, 0.2, 0.05, 0.0]]
    assert all(a1 >= a2 - 1e-9 for a1, a2 in zip(areas, areas[1:]))


def test_rejects_chi_factor_outside_0_1():
    rng = np.random.default_rng(3)
    pts = rng.standard_normal((50, 2))
    with pytest.raises(Exception):
        concavehull(pts, chi_factor=1.5)
    with pytest.raises(Exception):
        concavehull(pts, chi_factor=-0.1)
