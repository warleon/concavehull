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

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "point2d.hpp"  // Point2D

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

// The routing weight for one of `g`'s own edges, used by the generalized
// (Graph-accepting) find_bottleneck/find_interior_path/BottleneckSplit in
// place of Euclidean edge length -- another custom swappable callable
// (same reasoning as VertexDataFn/EdgeDataFn above). Defaults to
// euclidean_edge_weight everywhere it's an optional parameter, so omitting
// it reproduces the plain-geometry behavior exactly.
using EdgeWeightFn = std::function<double(const Graph&, size_t edge_idx)>;

inline double euclidean_edge_weight(const Graph& g, size_t edge_idx) {
	auto& [a, b] = g.edges[edge_idx];
	double dx = g.vertices[a].x - g.vertices[b].x;
	double dy = g.vertices[a].y - g.vertices[b].y;
	return std::sqrt(dx * dx + dy * dy);
}

// Single-source shortest-path distances from `source` to every vertex of
// `g`, using `weight_fn` as each edge's weight (Dijkstra; requires
// non-negative weights). Unreachable vertices get +infinity. Shared by the
// Graph-accepting find_bottleneck (split_criterion.hpp) for its
// generalized dist(A,B) -- run once per candidate source rather than once
// per (A,B) pair, since the paper's own Eq. 1-2 already needs the distance
// from each boundary vertex to every other one.
inline std::vector<double> dijkstra_from(const Graph& g, size_t source, const EdgeWeightFn& weight_fn) {
	size_t n = g.vertices.size();
	std::vector<std::vector<std::pair<size_t, size_t>>> adjacency(n);  // (neighbor, edge_idx)
	for (size_t e = 0; e < g.edges.size(); ++e) {
		auto& [a, b] = g.edges[e];
		adjacency[a].push_back({b, e});
		adjacency[b].push_back({a, e});
	}

	std::vector<double> dist(n, std::numeric_limits<double>::infinity());
	dist[source] = 0.0;
	using QueueItem = std::pair<double, size_t>;
	std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> open;
	open.push({0.0, source});
	std::vector<bool> closed(n, false);

	while (!open.empty()) {
		auto [d, u] = open.top();
		open.pop();
		if (closed[u]) continue;
		closed[u] = true;
		for (auto& [v, e] : adjacency[u]) {
			if (closed[v]) continue;
			double w = weight_fn(g, e);
			double nd = d + w;
			if (nd < dist[v]) {
				dist[v] = nd;
				open.push({nd, v});
			}
		}
	}
	return dist;
}
