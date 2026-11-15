"""Correctness tests for ObjectClusterGraph (the object-aware adaptation of
dtscan() for already-detected, individuated objects -- e.g. berries from an
ML model, each already its own mask) and GraphBottleneckSplit (the graph
-level generalization of BottleneckSplit: cuts a cluster of objects at its
narrowest graph connection instead of a polygon's narrowest boundary neck)."""

import numpy as np

from concavehull import GraphBottleneckCostCriterion, GraphBottleneckSplit, ObjectClusterGraph


def make_bunch_of_berries(rng, center, n_berries=10, spread=1.3, berry_radius=0.5, n_contour_pts=8):
    """One 'bunch': n_berries small circular contours clustered around center.

    Mirrors the real pipeline this is built for -- each berry is its own
    already-detected object (own mask/contour), several of which may be
    merged into one visual clump that needs separating back into bunches.
    """
    objects = []
    for _ in range(n_berries):
        bcx = center[0] + rng.normal(0, spread)
        bcy = center[1] + rng.normal(0, spread)
        theta = np.linspace(0, 2 * np.pi, n_contour_pts, endpoint=False)
        contour = np.stack([bcx + berry_radius * np.cos(theta), bcy + berry_radius * np.sin(theta)], axis=1)
        objects.append(contour)
    return objects


def run_full_pipeline(object_points, area_z_threshold=1.0, length_z_threshold=1.0, min_pts=3,
                       cost_threshold=0.1, min_group_fraction=0.1):
    ocg = ObjectClusterGraph(object_points, area_z_threshold, length_z_threshold, min_pts)
    clusters = ocg.clusters()
    gbs = GraphBottleneckSplit(GraphBottleneckCostCriterion(cost_threshold, min_group_fraction))
    final = []
    for c in clusters:
        if len(c) < 2:
            final.append(c)
            continue
        final.extend(gbs.split(c, ocg))
    return final


def test_three_well_separated_bunches_stay_clean_full_contour_mode():
    rng = np.random.default_rng(0)
    bunch_centers = [(0, 0), (14, 1), (6, 16)]
    object_points = []
    for c in bunch_centers:
        object_points.extend(make_bunch_of_berries(rng, c))

    final = run_full_pipeline(object_points)

    # exactly one group per true bunch, no spurious over-segmentation (the
    # failure this whole pipeline exists to fix: the old contour-concavity
    # -based BottleneckSplit produced dozens of spurious pieces from a single
    # bunch's own berry-shape noise -- here, each berry is one graph node
    # with no internal shape to be misread as a neck).
    assert len(final) == 3
    sizes = sorted(len(g) for g in final)
    assert sizes == [10, 10, 10]

    # every object assigned to exactly one final group
    all_assigned = sorted(i for g in final for i in g)
    assert all_assigned == list(range(30))


def test_three_well_separated_bunches_stay_clean_centroid_only_mode():
    rng = np.random.default_rng(0)
    bunch_centers = [(0, 0), (14, 1), (6, 16)]
    object_points = []
    for c in bunch_centers:
        for berry in make_bunch_of_berries(rng, c):
            object_points.append(berry.mean(axis=0, keepdims=True))  # centroid-only: size-1 point set

    final = run_full_pipeline(object_points)
    assert len(final) == 3
    assert sorted(len(g) for g in final) == [10, 10, 10]


def test_touching_bunches_merged_by_clustering_are_separated_by_graph_split():
    # Two bunches close enough that DTSCAN's own clustering merges them into
    # one cluster (the scenario GraphBottleneckSplit exists for) -- verified
    # directly that the initial clustering alone gives 1 cluster of 24, and
    # that best_normalized_cut (not a plain min cut, which this exact case
    # was used to catch: it picks a lopsided 3-vs-21 split instead of the
    # true 12-vs-12 one) is needed for the subsequent split to find the
    # correct bunch boundary. cost_threshold=0.1 (the library default) was
    # calibrated against best_normalized_cut's own cost scale: ~0.34 for an
    # intact single bunch's best (spurious) cut vs ~0.07 for this genuinely
    # merged pair's, verified directly outside this test.
    rng = np.random.default_rng(7)
    bunch_centers = [(0, 0), (5.5, 0)]
    object_points = []
    for c in bunch_centers:
        object_points.extend(make_bunch_of_berries(rng, c, n_berries=12))

    ocg = ObjectClusterGraph(object_points, 1.0, 1.0, 3)
    clusters = ocg.clusters()
    assert len(clusters) == 1
    assert len(clusters[0]) == 24

    gbs = GraphBottleneckSplit(GraphBottleneckCostCriterion(0.1, 0.1))
    sub = gbs.split(clusters[0], ocg)
    assert len(sub) == 2
    assert sorted(len(g) for g in sub) == [12, 12]

    # the two true bunches (objects 0-11 and 12-23) must not be mixed
    sub_sets = [set(g) for g in sub]
    true_a, true_b = set(range(12)), set(range(12, 24))
    assert {frozenset(s) for s in sub_sets} == {frozenset(true_a), frozenset(true_b)}


def test_graph_bottleneck_split_cost_is_scale_invariant():
    # Regression test for a real bug found during development: an earlier
    # version of GraphBottleneckCostCriterion normalized the min-cut's
    # affinity weight (1/distance) by a bare OBJECT COUNT, which is NOT
    # scale-invariant -- uniformly rescaling every coordinate shrinks every
    # affinity weight without changing any object count, so the exact same
    # relative geometry produced different split decisions depending purely
    # on absolute coordinate scale (e.g. small-unit synthetic test data vs.
    # real pixel-scale image coordinates). Confirmed directly: at one scale,
    # an intact single bunch was correctly left alone; at a 20x larger
    # (otherwise identical) scale, it was spuriously chopped into 7 pieces.
    # Fixed by normalizing against total affinity-weighted degree (Shi &
    # Malik's own normalized-cut denominator) instead, which cancels the
    # scale factor out of the ratio. This test asserts self-consistency
    # across a 40x range of absolute scale -- the same grouping decision at
    # every scale -- rather than hard-coding which specific decision is
    # "correct" for this one random bunch (whether a given borderline bunch
    # should or shouldn't split is a separate, threshold-calibration
    # question; scale must never be what decides it).
    bunch = make_bunch_of_berries(np.random.default_rng(1), (0, 0), n_berries=10)
    results = []
    for scale in (5.0, 20.0, 100.0, 200.0):
        scaled = [p * scale for p in bunch]
        final = run_full_pipeline(scaled)
        results.append(sorted(sorted(g) for g in final))
    assert all(r == results[0] for r in results)


def test_every_object_appears_in_exactly_one_final_group():
    rng = np.random.default_rng(5)
    bunch_centers = [(0, 0), (10, 2), (3, 12), (15, 13)]
    object_points = []
    for c in bunch_centers:
        object_points.extend(make_bunch_of_berries(rng, c, n_berries=7))

    final = run_full_pipeline(object_points)
    all_assigned = sorted(i for g in final for i in g)
    assert all_assigned == list(range(len(object_points)))


def test_graph_bottleneck_split_on_an_already_clean_group_is_a_no_op():
    rng = np.random.default_rng(0)
    object_points = make_bunch_of_berries(rng, (0, 0), n_berries=10)
    ocg = ObjectClusterGraph(object_points, 1.0, 1.0, 3)
    clusters = ocg.clusters()
    assert len(clusters) == 1

    gbs = GraphBottleneckSplit()
    sub = gbs.split(clusters[0], ocg)
    assert len(sub) == 1
    assert sorted(sub[0]) == list(range(10))
