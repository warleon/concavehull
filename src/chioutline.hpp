#pragma once

// Implementation of the chi-outline algorithm:
//
//   Zhong, X., & Duckham, M. (2016). Characterizing the shapes of noisy,
//   non-uniform, and disconnected point clusters in the plane. Computers,
//   Environment and Urban Systems, 57, 48-58.
//
// chi-outline extends the chi-shape algorithm (concavehull.hpp, Duckham et al.
// 2008) so that the output may be a set of disconnected simple polygons and
// may exclude outliers, while still using a single parameter chi. See
// "Algorithm 1" and Eqs. (2)-(4) of the paper for the pseudo code this file
// follows.
//
// Note on complexity: the paper describes an O(n log n) implementation via
// incremental bookkeeping of connected components. This implementation
// instead recomputes vertex degrees and connected components by direct
// (brute-force) scans of the triangulation whenever they are needed, which
// are operations that only occur O(n) times overall. That makes this
// implementation worst-case O(n^2) rather than O(n log n) -- a reasonable
// trade for a straightforward, easy-to-verify reference implementation.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "concavehull.hpp"
#include "thirdparty/delaunator-header-only.hpp"

class ChiOutline {
public:
	ChiOutline(const std::vector<double>& coords, double chi);

	// delaunator::Delaunator holds its coords argument by reference (bound to
	// m_coords_span below), so a ChiOutline must never be copied or moved --
	// doing so would silently leave that reference dangling at the old
	// address. Construct it in its final location (e.g. via std::make_unique)
	// instead.
	ChiOutline(const ChiOutline&) = delete;
	ChiOutline& operator=(const ChiOutline&) = delete;
	ChiOutline(ChiOutline&&) = delete;
	ChiOutline& operator=(ChiOutline&&) = delete;

	// Runs the algorithm and returns one polygon (as a flat [x0,y0,x1,y1,...]
	// coordinate vector, same convention as concavehull()) per disconnected
	// region in the final chi-outline. Outlier points are simply absent from
	// every returned polygon.
	std::vector<std::vector<double>> compute();

private:
	// NOTE on member order: `m_coords` must be constructed before
	// `m_coords_span`, which must be constructed before `d`, since
	// delaunator::Delaunator stores its `coords` argument *by reference*.
	// Keeping an owned `m_coords` vector (rather than referencing whatever
	// buffer the caller passed in) means a ChiOutline instance is safe to use
	// for as long as it exists, with no lifetime coupling to the caller's data.
	std::vector<double> m_coords;
	std::span<double> m_coords_span;
	delaunator::Delaunator d;

	double chi;
	size_t n_points;
	size_t n_triangles;

	std::vector<bool> removed_triangle;
	std::vector<bool> point_removed;
	std::vector<int> triangle_component;
	std::unordered_map<int, double> component_sigma;
	std::unordered_map<int, double> component_lambda;
	int next_component_id = 1;

	// Boundary B, kept as (length - lambda, halfedge) so the max element is
	// always the next edge to consider, plus an index for O(log n) removal by
	// halfedge id.
	std::multiset<std::pair<double, size_t>> boundary;
	std::unordered_map<size_t, std::multiset<std::pair<double, size_t>>::iterator> boundary_it;

	// Number of current boundary half-edges touching each point (at either
	// end). Used to test whether a point is currently on the boundary.
	std::vector<int> boundary_point_count;

	size_t triangle_of(size_t he) const { return he / 3; }

	// Is the edge `he` currently an exterior edge, i.e. is there no alive
	// triangle on its other side?
	bool is_exterior(size_t he) const {
		size_t twin = d.halfedges[he];
		if (twin == delaunator::INVALID_INDEX) return true;
		return removed_triangle[triangle_of(twin)];
	}

	double boundary_key(size_t he) {
		int comp = triangle_component[triangle_of(he)];
		return edge_length(d, he) - component_lambda.at(comp);
	}

	void insert_boundary(size_t he) {
		auto it = boundary.insert({boundary_key(he), he});
		boundary_it[he] = it;
		boundary_point_count[d.triangles[he]]++;
		boundary_point_count[d.triangles[next_halfedge(he)]]++;
	}

	void erase_boundary(size_t he) {
		auto found = boundary_it.find(he);
		if (found == boundary_it.end()) return;
		boundary.erase(found->second);
		boundary_it.erase(found);
		boundary_point_count[d.triangles[he]]--;
		boundary_point_count[d.triangles[next_halfedge(he)]]--;
	}

	bool point_on_boundary(size_t v) const { return boundary_point_count[v] > 0; }

	void rebuild_boundary_keys() {
		std::vector<size_t> edges;
		edges.reserve(boundary.size());
		for (auto& kv : boundary) edges.push_back(kv.second);
		boundary.clear();
		boundary_it.clear();
		for (size_t he : edges) {
			auto it = boundary.insert({boundary_key(he), he});
			boundary_it[he] = it;
		}
	}

	// Number of currently-alive triangles incident to point v. (Brute force;
	// see complexity note above.)
	size_t alive_degree(size_t v) const {
		size_t count = 0;
		for (size_t t = 0; t < n_triangles; ++t) {
			if (removed_triangle[t]) continue;
			if (d.triangles[3 * t] == v || d.triangles[3 * t + 1] == v || d.triangles[3 * t + 2] == v) {
				++count;
			}
		}
		return count;
	}

	// Mean/stddev of the lengths of the unique edges of the given triangles.
	std::pair<double, double> edge_stats(const std::vector<size_t>& tris) {
		std::vector<double> lengths;
		lengths.reserve(tris.size() * 3);
		for (size_t t : tris) {
			for (int k = 0; k < 3; ++k) {
				size_t he = 3 * t + k;
				size_t twin = d.halfedges[he];
				// count each undirected edge exactly once
				if (twin == delaunator::INVALID_INDEX || he < twin) {
					lengths.push_back(edge_length(d, he));
				}
			}
		}
		if (lengths.empty()) return {0.0, 0.0};
		double mean = 0.0;
		for (double l : lengths) mean += l;
		mean /= lengths.size();
		double var = 0.0;
		for (double l : lengths) var += (l - mean) * (l - mean);
		var /= lengths.size();
		return {mean, std::sqrt(var)};
	}

	// All alive triangles currently tagged `comp`, partitioned into connected
	// pieces (two triangles are adjacent if they share a live edge).
	std::vector<std::vector<size_t>> find_pieces(int comp) const {
		std::vector<size_t> members;
		for (size_t t = 0; t < n_triangles; ++t) {
			if (!removed_triangle[t] && triangle_component[t] == comp) members.push_back(t);
		}
		std::unordered_set<size_t> member_set(members.begin(), members.end());
		std::unordered_set<size_t> visited;
		std::vector<std::vector<size_t>> pieces;
		for (size_t start : members) {
			if (visited.count(start)) continue;
			std::vector<size_t> piece;
			std::vector<size_t> stack{start};
			visited.insert(start);
			while (!stack.empty()) {
				size_t t = stack.back();
				stack.pop_back();
				piece.push_back(t);
				for (int k = 0; k < 3; ++k) {
					size_t he = 3 * t + k;
					size_t twin = d.halfedges[he];
					if (twin == delaunator::INVALID_INDEX) continue;
					size_t nt = triangle_of(twin);
					if (member_set.count(nt) && !visited.count(nt)) {
						visited.insert(nt);
						stack.push_back(nt);
					}
				}
			}
			pieces.push_back(std::move(piece));
		}
		return pieces;
	}

	std::unordered_set<size_t> piece_points(const std::vector<size_t>& tris) const {
		std::unordered_set<size_t> pts;
		for (size_t t : tris) {
			pts.insert(d.triangles[3 * t]);
			pts.insert(d.triangles[3 * t + 1]);
			pts.insert(d.triangles[3 * t + 2]);
		}
		return pts;
	}

	size_t min_component_points() const {
		return (size_t)std::max(0LL, std::llround(std::log((double)n_points)));
	}

	// Permanently discard a piece: mark its triangles dead and purge any of
	// its edges from the boundary. Returns the set of points that lost their
	// last alive triangle as a result (these become outliers).
	void discard_piece(const std::vector<size_t>& tris) {
		for (size_t t : tris) {
			for (int k = 0; k < 3; ++k) {
				size_t he = 3 * t + k;
				erase_boundary(he);
			}
			removed_triangle[t] = true;
		}
		for (size_t v : piece_points(tris)) {
			if (alive_degree(v) == 0) point_removed[v] = true;
		}
	}

	// Rotate around vertex v (which is triangles[next_halfedge(incoming)]) from
	// a half-edge known to depart v within an *alive* triangle, returning the
	// next currently-exterior half-edge departing v. Used only while walking a
	// finished boundary to reconstruct output polygons.
	size_t rotate_to_next_boundary(size_t outgoing_candidate) const {
		size_t he = outgoing_candidate;
		size_t first = he;
		do {
			size_t t = triangle_of(he);
			if (removed_triangle[t]) return delaunator::INVALID_INDEX;
			if (is_exterior(he)) return he;
			size_t twin = d.halfedges[he];
			he = next_halfedge(twin);
		} while (he != first);
		return delaunator::INVALID_INDEX;
	}

	void create_root_component() {
		std::vector<size_t> all_tris(n_triangles);
		for (size_t t = 0; t < n_triangles; ++t) {
			triangle_component[t] = 0;
			all_tris[t] = t;
		}
		auto [mu, sigma] = edge_stats(all_tris);
		component_sigma[0] = sigma;
		component_lambda[0] = mu + chi * sigma;

		std::vector<size_t> hull_pts = get_hull_points(d);
		if (!hull_pts.empty()) hull_pts.pop_back();  // get_hull_points wraps the start point back on
		for (size_t p : hull_pts) {
			insert_boundary(d.hull_tri[p]);
		}
	}

	std::vector<std::vector<double>> extract_polygons() const {
		std::unordered_map<int, std::vector<size_t>> by_component;
		for (auto& kv : boundary) {
			size_t he = kv.second;
			int comp = triangle_component[triangle_of(he)];
			by_component[comp].push_back(he);
		}

		size_t min_points = min_component_points();
		std::unordered_set<size_t> used;
		std::vector<std::vector<double>> polygons;

		for (auto& kv : by_component) {
			const std::vector<size_t>& edges = kv.second;

			std::unordered_set<size_t> verts;
			for (size_t he : edges) {
				verts.insert(d.triangles[he]);
				verts.insert(d.triangles[next_halfedge(he)]);
			}
			if (verts.size() < min_points) continue;

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
					size_t nxt = rotate_to_next_boundary(next_halfedge(cur));
					if (nxt == delaunator::INVALID_INDEX || used.count(nxt)) {
						ok = false;
						break;
					}
					cur = nxt;
				}
				if (ok && poly.size() >= 6) polygons.push_back(std::move(poly));
			}
		}
		return polygons;
	}
};

inline ChiOutline::ChiOutline(const std::vector<double>& coords, double chi_)
	: m_coords(coords),
	  m_coords_span(m_coords),
	  d(m_coords_span),
	  chi(chi_),
	  n_points(m_coords.size() / 2),
	  n_triangles(d.triangles.size() / 3),
	  removed_triangle(n_triangles, false),
	  point_removed(n_points, false),
	  triangle_component(n_triangles, 0),
	  boundary_point_count(n_points, 0) {
	if (chi < 0) {
		throw std::invalid_argument("chi must be non-negative");
	}
	create_root_component();
}

inline std::vector<std::vector<double>> ChiOutline::compute() {
	while (!boundary.empty()) {
		auto top = *boundary.rbegin();
		double key = top.first;
		size_t e = top.second;

		if (key <= 0.0) break;  // L(B[1]) <= lambda[1]

		int comp = triangle_component[triangle_of(e)];
		size_t apex = get_interior_point(d, e);
		size_t a = d.triangles[e];
		size_t b = d.triangles[next_halfedge(e)];
		size_t he_bc = next_halfedge(e);  // b -> apex
		size_t he_ca = prev_halfedge(e);  // apex -> a

		bool bc_exterior = is_exterior(he_bc);
		bool ca_exterior = is_exterior(he_ca);
		bool isolating = bc_exterior || ca_exterior;
		bool apex_already_boundary = point_on_boundary(apex);

		erase_boundary(e);

		if (isolating && apex_already_boundary) {
			// --- isolating irregular exterior edge (Algorithm 1, lines 1.6-1.20) ---
			size_t apex_tri = triangle_of(e);

			if (bc_exterior) {
				erase_boundary(he_bc);
			} else {
				insert_boundary(d.halfedges[he_bc]);
			}
			if (ca_exterior) {
				erase_boundary(he_ca);
			} else {
				insert_boundary(d.halfedges[he_ca]);
			}

			for (size_t v : {a, b, apex}) {
				if (alive_degree(v) == 1) point_removed[v] = true;  // degree 2 before removal
			}
			removed_triangle[apex_tri] = true;

			std::vector<std::vector<size_t>> pieces = find_pieces(comp);
			if (pieces.size() > 1) {
				std::vector<std::vector<size_t>> retained;
				for (auto& piece : pieces) {
					if (piece_points(piece).size() < min_component_points()) {
						discard_piece(piece);
					} else {
						retained.push_back(std::move(piece));
					}
				}

				if (retained.size() >= 2) {
					component_sigma.erase(comp);
					component_lambda.erase(comp);

					std::vector<int> new_ids;
					for (auto& piece : retained) {
						int new_id = next_component_id++;
						for (size_t t : piece) triangle_component[t] = new_id;
						auto [mu, sigma] = edge_stats(piece);
						component_sigma[new_id] = sigma;
						component_lambda[new_id] = mu;  // mu stashed; lambda finalized below
						new_ids.push_back(new_id);
					}

					double sigma_bar = 0.0;
					for (auto& kv : component_sigma) sigma_bar += kv.second;
					sigma_bar /= component_sigma.size();

					for (int id : new_ids) {
						double mu = component_lambda[id];
						component_lambda[id] = mu + chi * sigma_bar;
					}

					rebuild_boundary_keys();
				}
				// if exactly one piece retained, it keeps the old `comp` id/lambda
				// unchanged -- nothing further to do.
			}
		} else {
			// --- regular, or irregular-but-not-isolating, exterior edge
			//     (Algorithm 1, lines 1.21-1.26) ---
			removed_triangle[triangle_of(e)] = true;
			insert_boundary(d.halfedges[he_bc]);
			insert_boundary(d.halfedges[he_ca]);
		}

		if (boundary.empty()) {
			return {};  // Algorithm 1, lines 1.27-1.29
		}
	}

	return extract_polygons();
}
