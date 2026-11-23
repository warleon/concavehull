#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "graph.hpp"
#include "thirdparty/delaunator-header-only.hpp"

inline size_t next_halfedge(size_t e) {
		return (e % 3 == 2) ? e - 2 : e + 1;
}

inline size_t prev_halfedge(size_t e) {
	return (e % 3 == 0) ? e + 2 : e - 1;
}

std::vector<size_t> get_hull_points(delaunator::Delaunator& d) {
	std::vector<size_t> hull_pts;

	size_t point = d.hull_start;
	do {
		hull_pts.push_back(point);
		point = d.hull_next[point];
	}
	while (point != d.hull_start);

	// Wrap back around
	hull_pts.push_back(d.hull_start);

	return hull_pts;
}

std::vector<double> get_hull_coords(delaunator::Delaunator& d) {
	const std::vector<size_t> hull_pts = get_hull_points(d);

	std::vector<double> hull_coords;
	hull_coords.reserve(2 * hull_pts.size());

	for (size_t point : hull_pts) {
		double x = d.coords[2 * point];
		double y = d.coords[2 * point + 1];
		hull_coords.push_back(x);
		hull_coords.push_back(y);
	}

	return hull_coords;
}

double edge_length(delaunator::Delaunator& d, size_t e_a) {
	size_t e_b = next_halfedge(e_a);

	double x_a = d.coords[2 * d.triangles[e_a]];
	double y_a = d.coords[2 * d.triangles[e_a] + 1];

	double x_b = d.coords[2 * d.triangles[e_b]];
	double y_b = d.coords[2 * d.triangles[e_b] + 1];

	return sqrt(pow(x_a - x_b, 2) + pow(y_a - y_b, 2));
}

size_t get_interior_point(delaunator::Delaunator& d, size_t e) {
	return d.triangles[next_halfedge(next_halfedge(e))];
}

// The chi-shape dig loop (Duckham et al. 2008), parameterized by an
// `on_absorb(triangle_index)` callback invoked exactly when a triangle gets
// carved OUT of the enclosed region (every time an interior point c is
// newly absorbed into the boundary, replacing edge (a,b) with (a,c)+(c,b),
// the triangle (a,b,c) -- identified by `triangle_of(e)` -- becomes
// exterior). concavehull() below calls this with a no-op callback
// (identical compiled behavior to the original, unparameterized version);
// concavehull_graph() uses the callback to know which triangles to exclude
// from "the complete graph" it returns.
template <typename OnAbsorb>
inline std::vector<double> concavehull_impl(delaunator::Delaunator& d, double chi_factor, OnAbsorb&& on_absorb) {
	// Determine initial points on outside hull
	std::vector<size_t> bpoints = get_hull_points(d);
	std::set<size_t> bset(bpoints.begin(), bpoints.end());

	// Make max heap of boundary edges with lengths
	typedef std::pair<size_t, double> hpair;
	
	auto cmp = [](const hpair& left, const hpair& right) {
		return left.second < right.second;
	};
	
	std::vector<hpair> bheap;
	bheap.reserve(bpoints.size());

	double max_len = std::numeric_limits<double>::min();
	double min_len = std::numeric_limits<double>::max();
	
	for (auto point : bpoints) {
		size_t e = d.hull_tri[point];
		double len = edge_length(d, e);

		bheap.push_back({e, len});
		std::push_heap(bheap.begin(), bheap.end(), cmp);

		min_len = std::min(len, min_len);
		max_len = std::max(len, max_len);
	}

	// Determine length parameter
	double length_param = chi_factor * max_len + (1 - chi_factor) * min_len;

	// Iteratively add points to boundary by iterating over the triangles on the hull
	while (!bheap.empty()) {

		// Get edge with the largest length
		std::pop_heap(bheap.begin(), bheap.end(), cmp);
		const auto [e, len] = bheap.back();
		bheap.pop_back();

		// Length of edge too small for our chi factor
		if (len <= length_param) {
			break;
		}

		// Find interior point given edge e (a -> b)
		//       e
		//  b <----- a
		//     \   /
		//  e_b \ / e_a
		//       c
		size_t c = get_interior_point(d, e);

		// Point already belongs to boundary
		if (bset.count(c)) {
			continue;
		}

		// Get two edges connected to interior point
		//  c -> b
		size_t e_b = d.halfedges[next_halfedge(e)];
		//  a -> c
		size_t e_a = d.halfedges[next_halfedge(next_halfedge(e))];

		// Add edges to heap
		double len_a = edge_length(d, e_a);
		double len_b = edge_length(d, e_b);

		bheap.push_back({e_a, len_a});
		std::push_heap(bheap.begin(), bheap.end(), cmp);
		bheap.push_back({e_b, len_b});
		std::push_heap(bheap.begin(), bheap.end(), cmp);
		
		// Update outer hull and connect new edges
		size_t a = d.triangles[e];
		size_t b = d.triangles[next_halfedge(e)];

		d.hull_next[c] = b;
		d.hull_prev[c] = a;
		d.hull_next[a] = d.hull_prev[b] = c;

		bset.insert(c);
		on_absorb(e / 3);  // triangle (a, b, c) is now excluded from the enclosed region
	}

	return get_hull_coords(d);
}

inline std::vector<double> concavehull(const std::span<double>& coords, double chi_factor = 0.1) {
	if (chi_factor < 0 || chi_factor > 1) {
		throw std::invalid_argument("Chi factor must be between 0 and 1 inclusive");
	}
	delaunator::Delaunator d(coords);
	return concavehull_impl(d, chi_factor, [](size_t) {});
}

// "The complete graph" for concavehull(): the Delaunay triangulation of
// every input point, MINUS every triangle the dig excluded (see
// concavehull_impl's own comment) -- not the raw unfiltered triangulation,
// which would make this independent of chi_factor and defeat the purpose.
// An edge survives iff at least one of its (one or two) adjacent triangles
// survived; the boundary edge (a,b) replaced by (a,c)+(c,b) at an
// absorption step correctly disappears too, since its only adjacent
// triangle was the one just excluded.
inline Graph concavehull_graph(const std::span<double>& coords, double chi_factor = 0.1) {
	if (chi_factor < 0 || chi_factor > 1) {
		throw std::invalid_argument("Chi factor must be between 0 and 1 inclusive");
	}
	delaunator::Delaunator d(coords);

	std::unordered_set<size_t> excluded_triangles;
	concavehull_impl(d, chi_factor, [&](size_t t) { excluded_triangles.insert(t); });

	size_t n_triangles = d.triangles.size() / 3;
	auto triangle_excluded = [&](size_t t) { return excluded_triangles.count(t) != 0; };

	Graph g;
	g.vertices.reserve(d.coords.size() / 2);
	for (size_t p = 0; p < d.coords.size() / 2; ++p) {
		g.vertices.push_back({d.coords[2 * p], d.coords[2 * p + 1]});
	}

	std::set<std::pair<size_t, size_t>> kept_edges;
	for (size_t t = 0; t < n_triangles; ++t) {
		if (triangle_excluded(t)) continue;
		for (int k = 0; k < 3; ++k) {
			size_t he = 3 * t + k;
			size_t pa = d.triangles[he];
			size_t pb = d.triangles[next_halfedge(he)];
			kept_edges.insert(pa < pb ? std::make_pair(pa, pb) : std::make_pair(pb, pa));
		}
	}
	g.edges.assign(kept_edges.begin(), kept_edges.end());
	return g;
}

