#pragma once

// A plain undirected graph over polygon/point-cloud vertices: the full
// surviving mesh of whichever outline algorithm produced it (concavehull(),
// ChiOutline, AlphaOutline -- see each's own *_graph()/compute_graph()
// method), not just its boundary cycle. Optionally annotated with custom
// per-vertex/per-edge data via a swappable callable (see annotate_graph
// below), which the generalized bottleneck-splitting machinery
// (bottleneck.hpp's BottleneckSplit::split_graph, interior_path.hpp's
// Graph-accepting find_interior_path, split_criterion.hpp's Graph-accepting
// find_bottleneck) can then use in place of Euclidean edge length/1.

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "split_criterion.hpp"  // Point2D

struct Graph {
	std::vector<Point2D> vertices;
	std::vector<std::pair<size_t, size_t>> edges;  // each undirected edge listed once
	std::vector<double> vertex_data;               // empty until annotated; else size() == vertices.size()
	std::vector<double> edge_data;                 // empty until annotated; else size() == edges.size()
};

// A custom swappable function/functor/callable (std::function accepts any
// of the three directly, and pybind11 accepts a plain Python callable here
// too) computing one value per vertex, given the graph's own structure.
using VertexDataFn = std::function<std::vector<double>(const Graph&)>;

// Same, for edges -- runs AFTER vertex_fn in annotate_graph(), so an edge
// callable may read the graph's just-populated vertex_data (e.g. to sum its
// two endpoints' values -- see edge_weight_sum_of_vertex_data in
// graph_callables.hpp).
using EdgeDataFn = std::function<std::vector<double>(const Graph&)>;

// Applies `vertex_fn` (if any) then `edge_fn` (if any) to `g`, in that
// order, storing their results into g.vertex_data/g.edge_data. Either may be
// an empty std::function (default-constructed/nullptr) to skip that half.
inline void annotate_graph(Graph& g, const VertexDataFn& vertex_fn, const EdgeDataFn& edge_fn) {
	if (vertex_fn) g.vertex_data = vertex_fn(g);
	if (edge_fn) g.edge_data = edge_fn(g);
}
