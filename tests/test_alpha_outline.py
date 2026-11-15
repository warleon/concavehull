"""Correctness tests for AlphaOutline -- the hole-capable, Delaunay-
triangulation-based outline algorithm (alpha-shape family / Polylidar-style),
as distinct from ChiOutline, which deliberately forces a single simple
(hole-free) polygon per region."""

import numpy as np
import pytest

from concavehull import AlphaOutline


def polygon_area(poly) -> float:
    poly = np.asarray(poly)
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def point_in_polygon(point, poly) -> bool:
    x, y = point
    poly = np.asarray(poly)
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


def ring_of_blobs(seed=0, n_ring=60, radius=5.0):
    # points scattered in a thin ring around an empty center -- a genuine
    # hole that ChiOutline cannot represent (see chioutline.hpp's own
    # count_boundary_cycles() comment: a second boundary cycle is treated as
    # "this component just split", not as "this component has a hole").
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(n_ring):
        theta = 2 * np.pi * i / n_ring
        for _ in range(6):
            r = radius + rng.normal(0, 0.15)
            pts.append(
                [r * np.cos(theta) + rng.normal(0, 0.15), r * np.sin(theta) + rng.normal(0, 0.15)]
            )
    return np.array(pts)


def test_ring_of_points_produces_a_hole():
    pts = ring_of_blobs()
    ao = AlphaOutline(pts, chi=1.0)
    polys = ao.compute()

    assert len(polys) == 1
    poly = polys[0]
    assert len(poly["holes"]) == 1

    exterior_area = polygon_area(poly["exterior"])
    hole_area = polygon_area(poly["holes"][0])
    # the hole should be a substantial fraction of the ring's own footprint,
    # not a sliver -- this is a real enclosed gap, not boundary noise.
    assert hole_area > 0.3 * exterior_area


def test_solid_blob_has_no_hole():
    # A specific, verified-by-hand (seed, chi) combination with zero spurious
    # holes. AlphaOutline uses a single GLOBAL edge-length threshold (unlike
    # ChiOutline's adaptive per-component one), so even a solid Gaussian blob
    # can occasionally get a small spurious interior hole purely from chance
    # density fluctuation -- a documented characteristic of alpha shapes in
    # general, not a bug (verified directly: chi=1.0 on this same family of
    # blobs reliably produces small spurious holes). This test pins one
    # specific clean case rather than asserting the (false) universal
    # property "a solid blob never has a hole".
    rng = np.random.default_rng(5)
    pts = rng.normal(0, 1.0, size=(300, 2))
    ao = AlphaOutline(pts, chi=0.3)
    polys = ao.compute()

    assert len(polys) == 1
    assert len(polys[0]["holes"]) == 0


def test_disconnected_components_stay_separate():
    rng = np.random.default_rng(2)
    centers = [(0, 0), (20, 0), (10, 20)]
    pts = np.vstack([rng.normal([cx, cy], 0.6, size=(100, 2)) for cx, cy in centers])
    ao = AlphaOutline(pts, chi=0.5)
    polys = ao.compute()
    assert len(polys) == 3


def test_every_exterior_vertex_and_hole_vertex_is_an_input_point():
    pts = ring_of_blobs(seed=1)
    ao = AlphaOutline(pts, chi=1.0)
    polys = ao.compute()
    assert len(polys) >= 1
    for poly in polys:
        for v in poly["exterior"]:
            assert np.any(np.all(np.isclose(pts, v), axis=1))
        for hole in poly["holes"]:
            for v in hole:
                assert np.any(np.all(np.isclose(pts, v), axis=1))


def test_hole_is_contained_within_its_own_exterior():
    pts = ring_of_blobs(seed=0)
    ao = AlphaOutline(pts, chi=1.0)
    polys = ao.compute()
    assert len(polys) == 1 and len(polys[0]["holes"]) == 1

    exterior = polys[0]["exterior"]
    hole = polys[0]["holes"][0]
    for v in hole:
        assert point_in_polygon(v, exterior) or any(np.allclose(v, e) for e in exterior)


def test_rejects_malformed_input():
    with pytest.raises(Exception):
        AlphaOutline(np.zeros((5, 3)), chi=0.1)  # wrong second dimension
