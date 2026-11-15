#pragma once

// AlphaOutline: a Delaunay-triangulation-based outline algorithm, in the
// same family as concavehull() (chi-shape, Duckham et al. 2008) and
// ChiOutline (chi-outline, Zhong & Duckham 2016), but explicitly KEEPS
// interior holes instead of forcing a single simple (hole-free) polygon per
// connected region.
//
// ChiOutline deliberately treats a component developing more than one
// boundary cycle as evidence that it just split into two separate pieces
// (see chioutline.hpp's own count_boundary_cycles() comment: "a simply
// -connected, hole-free piece has exactly one boundary cycle"). That is the
// right behavior for ChiOutline's own goal -- one simple polygon per
// disconnected input region -- but it means a genuine hole (e.g. objects
// arranged in a ring around an empty interior) can never be represented:
// ChiOutline's incremental edge-removal would instead eventually carve the
// ring apart into pieces rather than ever reporting an enclosed hole.
//
// This is the classic alpha-shape family instead (Edelsbrunner, Kirkpatrick
// & Seidel 1983), applied the same way a modern Delaunay-triangulation-based
// polygon extractor like Polylidar (Castagno & Atkins, 2020) does: triangulate,
// filter triangles by a threshold, connected-component the survivors, and
// boundary-follow each component keeping EVERY cycle it has -- the
// largest-area cycle is the exterior, every other cycle is a hole.
//
// This is a one-shot algorithm, deliberately simpler than ChiOutline's
// incremental machinery: there is no progressive re-threshold-and-split
// here, so none of ChiOutline's per-component bookkeeping (component ids,
// incremental sigma tracking, re-keying a priority queue) is needed. The
// trade-off is that the edge-length threshold (mu + chi*sigma, same formula
// and parameter name as ChiOutline, for familiarity) is estimated ONCE,
// globally, over every triangle edge -- unlike ChiOutline, this does not
// adapt to very different point densities in different regions of the same
// input within a single call. For that adaptive behavior, use ChiOutline
// (or the DTSCAN-style object_cluster.hpp) for the grouping decision, and
// reach for AlphaOutline specifically when a hole-capable *outline* of an
// already-decided group of points is what's wanted.
//
// Complexity: triangulation is O(n log n) (delaunator); filtering, the
// connected-components pass, and the boundary walk are each one O(n) pass
// given the triangulation -- the same O(n log n) total Polylidar reports for
// its own triangulate-filter-extract pipeline.

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "concavehull.hpp"  // next_halfedge, edge_length, delaunator::Delaunator
#include "polygon_with_holes.hpp"
#include "thirdparty/delaunator-header-only.hpp"

inline double alpha_outline_polygon_area(const std::vector<double>& flat) {
	size_t n = flat.size() / 2;
	double s = 0.0;
	for (size_t k = 0; k < n; ++k) {
		size_t k2 = (k + 1) % n;
		s += flat[2 * k] * flat[2 * k2 + 1] - flat[2 * k2] * flat[2 * k + 1];
	}
	return std::abs(s) / 2.0;
}

class AlphaOutline {
public:
	AlphaOutline(const std::vector<double>& coords, double chi);

	// delaunator::Delaunator holds its coords argument by reference (bound to
	// m_coords_span below), so an AlphaOutline must never be copied or moved
	// -- same restriction as ChiOutline, for the same reason.
	AlphaOutline(const AlphaOutline&) = delete;
	AlphaOutline& operator=(const AlphaOutline&) = delete;
	AlphaOutline(AlphaOutline&&) = delete;
	AlphaOutline& operator=(AlphaOutline&&) = delete;

	// Runs the algorithm and returns one PolygonWithHoles per disconnected
	// region. Outlier points (never part of any kept triangle, or whose
	// region falls below the small-component filter) are simply absent.
	std::vector<PolygonWithHoles> compute() const;

private:
	std::vector<double> m_coords;
	std::span<double> m_coords_span;
	delaunator::Delaunator d;

	double chi;
	size_t n_points;
	size_t n_triangles;

	std::vector<bool> removed_triangle;

	size_t triangle_of(size_t he) const { return he / 3; }

	bool is_exterior(size_t he) const {
		size_t twin = d.halfedges[he];
		if (twin == delaunator::INVALID_INDEX) return true;
		return removed_triangle[triangle_of(twin)];
	}

	// Same filter ChiOutline uses (Eq. 2): components smaller than log(n)
	// points are treated as noise, not real shapes.
	size_t min_component_points() const {
		return (size_t)std::max(0LL, std::llround(std::log((double)n_points)));
	}
};

inline AlphaOutline::AlphaOutline(const std::vector<double>& coords, double chi_)
	: m_coords(coords),
	  m_coords_span(m_coords),
	  d(m_coords_span),
	  chi(chi_),
	  n_points(m_coords.size() / 2),
	  n_triangles(d.triangles.size() / 3),
	  removed_triangle(n_triangles, false) {
	// chi may legitimately be negative (lambda = mu + chi*sigma has no sign
	// restriction) -- same as ChiOutline.
	std::vector<double> lengths;
	lengths.reserve(n_triangles * 3);
	for (size_t t = 0; t < n_triangles; ++t) {
		for (int k = 0; k < 3; ++k) {
			size_t he = 3 * t + k;
			size_t twin = d.halfedges[he];
			if (twin == delaunator::INVALID_INDEX || he < twin) {
				lengths.push_back(edge_length(d, he));
			}
		}
	}
	double mu = 0.0;
	for (double l : lengths) mu += l;
	mu /= std::max<size_t>(lengths.size(), 1);
	double var = 0.0;
	for (double l : lengths) var += (l - mu) * (l - mu);
	var /= std::max<size_t>(lengths.size(), 1);
	double sigma = std::sqrt(var);
	double lambda = mu + chi * sigma;

	for (size_t t = 0; t < n_triangles; ++t) {
		double max_edge = 0.0;
		for (int k = 0; k < 3; ++k) {
			max_edge = std::max(max_edge, edge_length(d, 3 * t + k));
		}
		removed_triangle[t] = max_edge > lambda;
	}
}

inline std::vector<PolygonWithHoles> AlphaOutline::compute() const {
	// Union-find over kept triangles, adjacent via a shared non-removed edge.
	std::vector<size_t> parent(n_triangles);
	std::iota(parent.begin(), parent.end(), 0);
	std::function<size_t(size_t)> find = [&](size_t x) {
		while (parent[x] != x) {
			parent[x] = parent[parent[x]];
			x = parent[x];
		}
		return x;
	};
	auto unite = [&](size_t a, size_t b) {
		a = find(a);
		b = find(b);
		if (a != b) parent[a] = b;
	};

	for (size_t t = 0; t < n_triangles; ++t) {
		if (removed_triangle[t]) continue;
		for (int k = 0; k < 3; ++k) {
			size_t he = 3 * t + k;
			size_t twin = d.halfedges[he];
			if (twin == delaunator::INVALID_INDEX) continue;
			size_t nt = triangle_of(twin);
			if (!removed_triangle[nt]) unite(t, nt);
		}
	}

	std::unordered_map<size_t, std::vector<size_t>> boundary_by_component;
	for (size_t t = 0; t < n_triangles; ++t) {
		if (removed_triangle[t]) continue;
		for (int k = 0; k < 3; ++k) {
			size_t he = 3 * t + k;
			if (is_exterior(he)) {
				boundary_by_component[find(t)].push_back(he);
			}
		}
	}

	size_t min_points = min_component_points();
	std::vector<PolygonWithHoles> out;

	for (auto& kv : boundary_by_component) {
		const std::vector<size_t>& edges = kv.second;
		std::unordered_set<size_t> used;
		std::vector<std::vector<double>> cycles;

		for (size_t start : edges) {
			if (used.count(start)) continue;
			std::vector<double> poly;
			size_t cur = start;
			size_t loop_start_point = d.triangles[start];
			bool ok = true;
			while (true) {
				used.insert(cur);
				size_t p = d.triangles[cur];
				poly.push_back(d.coords[2 * p]);
				poly.push_back(d.coords[2 * p + 1]);
				size_t next_point = d.triangles[next_halfedge(cur)];
				if (next_point == loop_start_point) break;

				size_t nxt = delaunator::INVALID_INDEX;
				size_t he = next_halfedge(cur);
				size_t first = he;
				do {
					size_t t = triangle_of(he);
					if (removed_triangle[t]) {
						nxt = delaunator::INVALID_INDEX;
						break;
					}
					if (is_exterior(he)) {
						nxt = he;
						break;
					}
					size_t twin = d.halfedges[he];
					he = next_halfedge(twin);
				} while (he != first);

				if (nxt == delaunator::INVALID_INDEX || used.count(nxt)) {
					ok = false;
					break;
				}
				cur = nxt;
			}
			if (ok && poly.size() / 2 >= 3) cycles.push_back(std::move(poly));
		}

		if (cycles.empty()) continue;

		size_t exterior_idx = 0;
		double best_area = -1.0;
		for (size_t i = 0; i < cycles.size(); ++i) {
			double a = alpha_outline_polygon_area(cycles[i]);
			if (a > best_area) {
				best_area = a;
				exterior_idx = i;
			}
		}

		size_t total_points = 0;
		for (auto& c : cycles) total_points += c.size() / 2;
		if (total_points < min_points) continue;

		PolygonWithHoles pwh;
		pwh.exterior = std::move(cycles[exterior_idx]);
		for (size_t i = 0; i < cycles.size(); ++i) {
			if (i != exterior_idx) pwh.holes.push_back(std::move(cycles[i]));
		}
		out.push_back(std::move(pwh));
	}

	return out;
}
