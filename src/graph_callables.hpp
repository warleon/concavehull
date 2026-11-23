#pragma once

// Concrete VertexDataFn/EdgeDataFn callables (graph.hpp) for the "weight a
// vertex by how connected it is, then weight each edge by its endpoints'
// combined connectedness" scheme: vertex_degree assigns each vertex its own
// incident-edge count; edge_weight_sum_of_vertex_data assigns each edge the
// sum of its two endpoints' (already-populated) vertex_data. Composed via
// annotate_graph(g, vertex_degree, edge_weight_sum_of_vertex_data), the
// resulting edge_data is then usable directly as an EdgeWeightFn
// (split_criterion.hpp/interior_path.hpp/bottleneck.hpp) -- i.e.
// `[](const Graph& g, size_t e) { return g.edge_data[e]; }` -- in place of
// Euclidean edge length for both bottleneck-cost and A* cut-path routing.

#include <cstddef>
#include <vector>

#include "graph.hpp"

inline std::vector<double> vertex_degree(const Graph& g) {
	std::vector<double> degree(g.vertices.size(), 0.0);
	for (auto& [a, b] : g.edges) {
		degree[a] += 1.0;
		degree[b] += 1.0;
	}
	return degree;
}

inline std::vector<double> edge_weight_sum_of_vertex_data(const Graph& g) {
	std::vector<double> weight(g.edges.size());
	for (size_t i = 0; i < g.edges.size(); ++i) {
		auto& [a, b] = g.edges[i];
		weight[i] = g.vertex_data[a] + g.vertex_data[b];
	}
	return weight;
}
