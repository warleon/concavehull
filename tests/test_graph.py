"""Correctness tests for Graph, annotate_graph, the vertex_degree/
edge_weight_sum_of_vertex_data callable pair, and the "complete graph"
output added to concavehull_graph()/ChiOutline.compute_graph()/
AlphaOutline.compute_graph()."""

import numpy as np
import pytest

from concavehull import (
    AlphaOutline,
    ChiOutline,
    Graph,
    annotate_graph,
    concavehull_graph,
    edge_weight_sum_of_vertex_data,
    vertex_degree,
)


def star_graph():
    # vertex 0 connected to 1,2,3,4; plus one extra edge 1-2.
    # degrees: v0=4, v1=2, v2=2, v3=1, v4=1.
    return Graph(
        vertices=np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float),
        edges=np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2]], dtype=np.int64),
    )


def test_hand_built_graph_roundtrips_vertices_and_edges():
    g = star_graph()
    assert g.vertices.shape == (5, 2)
    assert np.allclose(g.vertices[0], [0, 0])
    assert g.edges.shape == (5, 2)
    assert g.vertex_data is None
    assert g.edge_data is None


def test_annotate_graph_with_builtin_callables():
    g = star_graph()
    annotate_graph(g, vertex_degree, edge_weight_sum_of_vertex_data)

    assert np.allclose(g.vertex_data, [4, 2, 2, 1, 1])

    expected_edge_weight = {
        (0, 1): 6,
        (0, 2): 6,
        (0, 3): 5,
        (0, 4): 5,
        (1, 2): 4,
    }
    for (a, b), w in zip(g.edges, g.edge_data):
        assert w == expected_edge_weight[(int(a), int(b))]


def test_annotate_graph_with_custom_python_callable():
    g = Graph(
        vertices=np.array([[0, 0], [1, 0], [0, 1]], dtype=float),
        edges=np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64),
    )
    annotate_graph(
        g,
        lambda gr: [7.0] * len(gr.vertices),
        lambda gr: [gr.vertex_data[a] + gr.vertex_data[b] for a, b in gr.edges],
    )
    assert np.allclose(g.vertex_data, [7, 7, 7])
    assert np.allclose(g.edge_data, [14, 14, 14])


def test_annotate_graph_skips_none_callables():
    g = star_graph()
    annotate_graph(g, vertex_degree, None)
    assert g.vertex_data is not None
    assert g.edge_data is None


def test_concavehull_graph_never_drops_a_point():
    # concavehull()'s dig never excludes a point from the final shape (only
    # triangles) -- so concavehull_graph()'s vertex count should always
    # equal the input point count, regardless of chi_factor.
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 1.0, size=(80, 2))
    for chi_factor in (0.05, 0.3, 1.0):
        g = concavehull_graph(pts, chi_factor)
        assert len(g.vertices) == len(pts)


def test_concavehull_graph_edge_count_depends_on_chi_factor():
    # Verified directly (not assumed) while building this: more digging
    # (lower chi_factor) excludes more triangles, so strictly fewer edges
    # survive -- confirming this is NOT just the raw unfiltered Delaunay
    # triangulation (which would be identical at every chi_factor).
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 1.0, size=(200, 2))
    n_edges_low = len(concavehull_graph(pts, 0.05).edges)
    n_edges_high = len(concavehull_graph(pts, 1.0).edges)
    assert n_edges_low < n_edges_high


def test_chioutline_compute_graph_region_count_matches_compute():
    # Regression test for a real bug found during development: a single
    # triangle_component id does not always mean one connected region
    # (find_pieces's own split-detection can disagree and leave one id
    # secretly hosting two disjoint triangle groups) -- grouping Graph
    # output by raw id alone silently merged such pairs into one Graph.
    # Checked across a disconnected multi-blob-with-outliers scene, where
    # this was originally caught.
    rng = np.random.default_rng(2)
    centers = [(0, 0), (20, 0), (10, 20), (25, 22)]
    pts = np.vstack([rng.normal(c, 0.6, size=(50, 2)) for c in centers])
    outliers = rng.uniform([-10, -10], [35, 35], size=(10, 2))
    pts = np.vstack([pts, outliers])

    co = ChiOutline(pts, 0.1)
    polys = co.compute()
    graphs = co.compute_graph()
    assert len(polys) == len(graphs)
    for poly, g in zip(sorted(polys, key=len), sorted(graphs, key=lambda gr: len(gr.vertices))):
        assert len(g.vertices) >= len(poly)


def test_alpha_outline_compute_graph_region_count_matches_compute():
    rng = np.random.default_rng(2)
    centers = [(0, 0), (20, 0), (10, 20)]
    pts = np.vstack([rng.normal(c, 0.6, size=(80, 2)) for c in centers])

    ao = AlphaOutline(pts, 0.5)
    polys = ao.compute()
    graphs = ao.compute_graph()
    assert len(polys) == len(graphs)


def test_alpha_outline_compute_graph_includes_interior_mesh_for_a_hole():
    rng = np.random.default_rng(0)
    pts = []
    n_ring = 60
    for i in range(n_ring):
        theta = 2 * np.pi * i / n_ring
        for _ in range(6):
            r = 5.0 + rng.normal(0, 0.15)
            pts.append([r * np.cos(theta) + rng.normal(0, 0.15), r * np.sin(theta) + rng.normal(0, 0.15)])
    pts = np.array(pts)

    ao = AlphaOutline(pts, 1.0)
    polys = ao.compute()
    graphs = ao.compute_graph()
    assert len(polys) == 1
    assert len(polys[0]["holes"]) == 1

    exterior_and_hole_verts = len(polys[0]["exterior"]) + sum(len(h) for h in polys[0]["holes"])
    assert len(graphs[0].vertices) >= exterior_and_hole_verts


def test_rejects_mismatched_vertex_edge_shapes():
    with pytest.raises(Exception):
        Graph(vertices=np.zeros((5, 3)), edges=np.array([[0, 1]], dtype=np.int64))
