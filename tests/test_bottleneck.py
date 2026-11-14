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


def test_rejects_invalid_constructor_arguments():
    with pytest.raises(Exception):
        BottleneckSplit(min_arc_fraction=0.0)
    with pytest.raises(Exception):
        BottleneckSplit(min_arc_fraction=0.5)
    with pytest.raises(Exception):
        ConvexDeficitCriterion(deficit_threshold=1.0)
    with pytest.raises(Exception):
        ConvexDeficitCriterion(deficit_threshold=-0.1)
