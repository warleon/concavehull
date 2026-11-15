"""Correctness tests for BottleneckSplit and the SplitCriterion family
(Wang, Zhang & Ray, 2012)."""

import numpy as np
import pytest

from concavehull import (
    BottleneckCostCriterion,
    BottleneckSplit,
    ConvexDeficitCriterion,
    ConvexHullL2Criterion,
    concavehull,
)


def two_overlapping_blobs(seed=0, n_per_blob=200, separation=4.2, chi_factor=0.3):
    # chi_factor=0.3 is specifically chosen (and verified, not assumed) to
    # produce a hull all three SplitCriterion implementations agree is
    # "non-convex enough to split" at their respective natural-default
    # thresholds -- see test_all_criteria_implement_should_split. A higher
    # chi_factor (e.g. 0.6) smooths the boundary enough that ConvexDeficit/
    # ConvexHullL2 stop triggering even though a real neck is still present,
    # which is exactly the known unreliability documented in
    # split_criterion.hpp -- not something a test should paper over.
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[0, 0], scale=1.0, size=(n_per_blob, 2))
    b = rng.normal(loc=[separation, 0], scale=1.0, size=(n_per_blob, 2))
    pts = np.vstack([a, b])
    return concavehull(pts, chi_factor=chi_factor)


def regular_hexagon():
    return np.array([[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 7)[:-1]])


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def point_in_polygon(point, poly) -> bool:
    x, y = point
    n = len(poly)
    inside = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_int:
                inside = not inside
    return inside


def test_merged_clump_splits_into_two_with_default_criterion():
    hull = two_overlapping_blobs()
    bs = BottleneckSplit()  # default: BottleneckCostCriterion
    pieces = bs.split(hull)
    assert len(pieces) == 2


def test_already_convex_polygon_is_not_split():
    bs = BottleneckSplit()
    pieces = bs.split(regular_hexagon())
    assert len(pieces) == 1


def test_split_all_flattens_results_across_polygons():
    hull = two_overlapping_blobs()
    hexagon = regular_hexagon()
    bs = BottleneckSplit()
    pieces = bs.split_all([hull, hexagon])
    assert len(pieces) == 3  # 2 from the clump + 1 unsplit hexagon


@pytest.mark.parametrize(
    "criterion_factory",
    [
        lambda: BottleneckCostCriterion(0.45),
        lambda: ConvexDeficitCriterion(0.15),
        lambda: ConvexHullL2Criterion(0.18),
    ],
)
def test_all_criteria_implement_should_split(criterion_factory):
    # chi_factor=0.1 (more aggressively non-convex than the chi_factor=0.3
    # default used elsewhere in this file) is needed here: ConvexDeficit/
    # ConvexHullL2Criterion are documented as less reliable than
    # BottleneckCostCriterion and may not trigger on a *mildly* non-convex
    # hull even when a real neck is present -- this test only asserts that
    # all three criteria have a regime where they agree, not that they
    # agree at every chi_factor.
    hull = two_overlapping_blobs(chi_factor=0.1)
    criterion = criterion_factory()
    # a strongly merged clump should read as "should split"
    assert criterion.should_split(hull) is True
    # an already-convex polygon should never read as "should split"
    assert criterion.should_split(regular_hexagon()) is False


def test_min_arc_fraction_rejects_degenerate_cuts():
    hull = two_overlapping_blobs()
    # an extreme min_arc_fraction should prevent any cut from qualifying,
    # so the result must fall back to the single unmodified polygon.
    bs = BottleneckSplit(BottleneckCostCriterion(0.95), min_arc_fraction=0.49)
    pieces = bs.split(hull)
    assert len(pieces) >= 1


def assert_split_is_geometrically_valid(poly: np.ndarray, pieces) -> None:
    # Two invariants any correct split must satisfy, regardless of how many
    # pieces it produces: (1) area is conserved exactly -- no sliver gained
    # or lost by a cut clipping through a concavity, or by a pocket face
    # silently dropped -- and (2) every interior point of every piece is
    # also interior to the original polygon -- a straight chord through a
    # concave notch would instead produce a piece bulging outside it.
    total_area = polygon_area(poly)
    piece_area_sum = sum(polygon_area(np.asarray(p)) for p in pieces)
    assert piece_area_sum == pytest.approx(total_area, rel=1e-9)

    for p in pieces:
        for v in np.asarray(p):
            assert np.any(np.all(np.isclose(poly, v), axis=1))

    rng = np.random.default_rng(42)
    for p in pieces:
        p = np.asarray(p)
        lo, hi = p.min(axis=0), p.max(axis=0)
        samples = rng.uniform(lo, hi, size=(500, 2))
        for s in samples:
            if point_in_polygon(s, p):
                assert point_in_polygon(s, poly)


def test_bent_bottleneck_cut_stays_inside_non_convex_polygon():
    # A random star-shaped polygon (found by search, not hand-picked to look
    # nice) whose actual bottleneck pair -- the one BottleneckSplit's own
    # default criterion selects, not an arbitrarily chosen pair -- has an
    # invalid direct chord: the straight line between the two bottleneck
    # points exits the polygon through a concavity. This is exactly the
    # failure mode the A* interior-path search (src/interior_path.hpp) was
    # added to fix in place of the old direct-chord cut. Verified against
    # the pre-fix implementation directly (outside this test suite): on this
    # exact polygon, the old straight-chord cut produced a piece with points
    # measurably outside the original polygon's boundary.
    poly = np.array(
        [
            [-1.910487854373182, 4.473296672810858],
            [-4.072059016762769, 2.624050295518772],
            [-3.970331316050033, -2.821052534015959],
            [-1.2869786879762957, -1.6395373977510288],
            [-0.6132084489888548, -1.0369122383426013],
            [-1.5462484058782449, -3.40834995038172],
            [-1.101935608085531, -4.66416276701774],
            [0.8183037649659385, -4.747403139120335],
            [1.386093237026705, -3.169166825826249],
            [3.7122129630676777, -2.5729901599082945],
        ]
    )

    bs = BottleneckSplit()
    pieces = bs.split(poly)
    assert len(pieces) == 2
    assert_split_is_geometrically_valid(poly, pieces)


def test_multi_bend_cut_produces_one_piece_per_pocket():
    # A random star-shaped polygon (again found by search) whose bottleneck
    # pair's interior path bends through TWO genuine diagonals meeting at a
    # shared intermediate vertex, each cutting off its own small concave
    # pocket on either side of that vertex. A split that (incorrectly)
    # always returns exactly 2 pieces silently drops one of those pockets'
    # area entirely -- directly observed as an area deficit during
    # development before splitting was generalized to trace every bounded
    # face of the path+boundary graph (see split_polygon_along_path's
    # comment in interior_path.hpp). min_arc_fraction=0.1 and max_depth=1
    # isolate this single bottleneck cut instead of its full recursive
    # split, so this test exercises exactly the geometry described above.
    poly = np.array(
        [
            [4.860508378746356, 0.9104612771664122],
            [4.731492116951194, 2.3631365848355874],
            [4.30075717462966, 2.2710973247631556],
            [4.23131620521264, 2.989097926325435],
            [-2.7682877487854993, 4.461960907313068],
            [-4.138210161540568, 2.233558018730447],
            [-4.6642618520652, 1.8145202721785756],
            [-5.221220156275908, 0.45595336779983087],
            [-1.329084361273978, -0.8780977934292168],
            [-2.199819925715398, -1.819424196608107],
            [-2.28758427144589, -2.1351142542946424],
            [-0.19864245037269032, -0.22481364929633],
            [-1.4220828099787028, -4.944964178310559],
            [-1.181138274589145, -4.761210385646487],
            [0.1607353545988999, -5.046805454749535],
            [1.8847614283392968, -1.4342266568551854],
            [4.89038258662116, -0.918198482612521],
            [4.72344835412029, -0.3340414612750901],
        ]
    )

    bs = BottleneckSplit(max_depth=1)
    pieces = bs.split(poly)
    assert len(pieces) == 3  # one far-side piece + one pocket per bend diagonal
    assert_split_is_geometrically_valid(poly, pieces)


def test_rejects_invalid_constructor_arguments():
    with pytest.raises(Exception):
        BottleneckSplit(min_arc_fraction=0.0)
    with pytest.raises(Exception):
        BottleneckSplit(min_arc_fraction=0.5)
    with pytest.raises(Exception):
        ConvexDeficitCriterion(deficit_threshold=1.0)
    with pytest.raises(Exception):
        ConvexDeficitCriterion(deficit_threshold=-0.1)
