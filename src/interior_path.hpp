#pragma once

// The weighted-shortest-path cut between two already-selected bottleneck
// points, per step 3 of:
//
//   Wang, H., Zhang, H., & Ray, N. (2012). Clump splitting via bottleneck
//   detection and shape classification. Pattern Recognition, 45(7), 2780-2787.
//
// The paper finds the optimal cut between points A* and B* via a weighted
// shortest path over image-intensity edge strength (Eqs. 3-4), by Dijkstra.
// This project has no source image, only polygon geometry, so there is no
// intensity information to route the cut around -- instead, this header
// treats the polygon's own geometry as the thing being routed through:
//
//   1. Triangulate the polygon's interior (ear clipping), giving a planar
//      graph whose nodes are the polygon's own vertices and whose edges are
//      the original boundary edges plus the triangulation's diagonals, each
//      weighted by its own Euclidean length.
//   2. Run A* (admissible/consistent heuristic: straight-line distance to
//      the goal, which is always a lower bound on any path's remaining
//      length) from A* to B* over that graph.
//   3. Split the polygon by tracing every bounded face of the planar graph
//      formed by the polygon's own boundary cycle plus the path's edges
//      (see split_polygon_along_path below). Usually that's exactly 2
//      pieces, but a path that bends through k >= 2 genuine diagonals
//      yields 1 + k pieces -- see that function's own comment.
//
// The resulting path is a cut that always stays inside the polygon -- unlike
// a direct chord, which for a non-convex polygon can leave the polygon
// through a concavity between the two points. Whenever the direct chord IS
// already a valid interior diagonal, find_interior_path() adds it to the
// graph explicitly (see is_valid_diagonal below), so A* finds it as the
// (tied-for-)shortest path and the result degenerates to exactly the old
// straight-chord behavior -- bending only happens when the straight line
// genuinely isn't available.
//
// Why face-tracing, not "arc + path" concatenation: an earlier version of
// this file built each output piece as [the boundary arc from i to j] +
// [the path's intermediate vertices, reversed] -- appended to BOTH pieces,
// on the reasoning that a cut becomes shared boundary between the two
// pieces it separates. That reasoning is correct for a path through
// genuinely NEW (interior, non-boundary) points, but every intermediate
// vertex here is necessarily one of the polygon's OWN existing vertices
// (there are no Steiner points), and every such vertex already belongs to
// one of the two boundary arcs. Appending it AGAIN to that same arc's piece
// produced a polygon with a duplicated vertex -- not just cosmetically odd,
// but something that then recursed pathologically (directly observed: a
// hand-built notch-shaped test polygon produced 43 output pieces, mostly
// degenerate slivers, before this fix). Restricting which vertices the path
// may reuse only moves the same failure to the other piece.
//
// The structurally correct fix is to never decide per-piece membership
// ad hoc at all: treat the polygon's boundary cycle plus the path's edges
// as one small planar graph, and extract every one of its bounded faces by
// the standard "always turn to the next edge in clockwise order from the
// reverse bearing" face-tracing rule (the same principle used to walk faces
// of any planar straight-line graph, e.g. a half-edge structure's own
// next-edge convention). This correctly handles a path that reuses existing
// vertices as bend points -- an unavoidable situation for any
// polygon-vertices-only geodesic path, not a rare edge case -- without ever
// producing a crossing or duplicated-not-as-a-genuine-pinch edge. A
// genuinely unavoidable pinch (the path touches the boundary at a vertex
// that is an articulation point of the resulting subdivision) still yields
// a geometrically valid, self-touching-at-one-point polygon, which is the
// correct topological answer for that case -- not a bug.
//
// Extracting "the faces" (plural, not a fixed two) matters as soon as a
// bent path crosses through more than one genuine diagonal: each such
// diagonal splits off its own small pocket face in addition to the one
// large far-side face, so the result can legitimately be 3+ pieces, not
// always 2 -- see split_polygon_along_path's own comment for why.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "graph.hpp"  // Graph, EdgeWeightFn, euclidean_edge_weight
#include "split_criterion.hpp"  // Point2D, distance()

inline double interior_path_cross(const Point2D& o, const Point2D& a, const Point2D& b) {
	return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

// Standard ray-casting point-in-simple-polygon test.
inline bool point_in_polygon(const Point2D& p, const std::vector<Point2D>& pts) {
	bool inside = false;
	size_t n = pts.size();
	for (size_t i = 0, j = n - 1; i < n; j = i++) {
		const Point2D& pi = pts[i];
		const Point2D& pj = pts[j];
		bool straddles = (pi.y > p.y) != (pj.y > p.y);
		if (straddles && p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x) {
			inside = !inside;
		}
	}
	return inside;
}

// True if segments (p1,p2) and (p3,p4) cross at an interior point of both
// (shared endpoints / collinear touching are NOT reported as crossing --
// is_valid_diagonal's midpoint-in-polygon check, combined with this,
// catches collinear-overlap cases correctly in practice for the simple,
// non-self-intersecting polygons this operates on).
inline bool segments_properly_intersect(const Point2D& p1, const Point2D& p2, const Point2D& p3, const Point2D& p4) {
	double d1 = interior_path_cross(p3, p4, p1);
	double d2 = interior_path_cross(p3, p4, p2);
	double d3 = interior_path_cross(p1, p2, p3);
	double d4 = interior_path_cross(p1, p2, p4);
	bool straddle_34 = (d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0);
	bool straddle_12 = (d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0);
	return straddle_34 && straddle_12;
}

// Is the chord pts[i]-pts[j] a valid interior diagonal of the simple polygon
// pts -- i.e. does it lie entirely inside the polygon, crossing no boundary
// edge? Checked as: (a) it doesn't properly cross any boundary edge not
// incident to i or j, and (b) its midpoint is inside the polygon -- (a)+(b)
// together are sufficient, since a segment can only move between inside and
// outside the polygon by crossing its boundary.
inline bool is_valid_diagonal(const std::vector<Point2D>& pts, size_t i, size_t j) {
	size_t n = pts.size();
	for (size_t k = 0; k < n; ++k) {
		size_t k2 = (k + 1) % n;
		if (k == i || k == j || k2 == i || k2 == j) continue;
		if (segments_properly_intersect(pts[i], pts[j], pts[k], pts[k2])) return false;
	}
	Point2D mid{(pts[i].x + pts[j].x) / 2.0, (pts[i].y + pts[j].y) / 2.0};
	return point_in_polygon(mid, pts);
}

inline bool point_in_or_on_triangle(const Point2D& p, const Point2D& a, const Point2D& b, const Point2D& c) {
	double d1 = interior_path_cross(a, b, p);
	double d2 = interior_path_cross(b, c, p);
	double d3 = interior_path_cross(c, a, p);
	bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
	bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
	return !(has_neg && has_pos);
}

// Ear-clipping triangulation of a simple polygon (handles non-convex, but
// not self-intersecting, polygons -- which is all this codebase ever
// produces). Returns the diagonal edges used (as index pairs into `pts`);
// `ok` is false if clipping stalled (e.g. from near-degenerate/duplicate
// points) before reaching a triangle, in which case `diagonals` may be
// incomplete and the caller should fall back to a direct chord.
struct EarClipResult {
	std::vector<std::pair<size_t, size_t>> diagonals;
	bool ok;
};

inline EarClipResult ear_clip_diagonals(const std::vector<Point2D>& pts) {
	size_t n = pts.size();
	if (n < 3) return {{}, false};

	double signed_area2 = 0.0;
	for (size_t i = 0; i < n; ++i) {
		size_t j = (i + 1) % n;
		signed_area2 += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
	}
	double orientation_sign = signed_area2 >= 0.0 ? 1.0 : -1.0;

	std::vector<size_t> next_idx(n), prev_idx(n);
	for (size_t i = 0; i < n; ++i) {
		next_idx[i] = (i + 1) % n;
		prev_idx[i] = (i + n - 1) % n;
	}

	std::vector<std::pair<size_t, size_t>> diagonals;
	diagonals.reserve(n - 3);
	size_t remaining = n;
	size_t current = 0;
	size_t steps_since_clip = 0;

	while (remaining > 3) {
		size_t a = prev_idx[current];
		size_t b = current;
		size_t c = next_idx[current];

		bool is_convex = orientation_sign * interior_path_cross(pts[a], pts[b], pts[c]) > 1e-12;
		bool ear = is_convex;
		if (ear) {
			for (size_t k = next_idx[c]; k != a; k = next_idx[k]) {
				if (point_in_or_on_triangle(pts[k], pts[a], pts[b], pts[c])) {
					ear = false;
					break;
				}
			}
		}

		if (ear) {
			diagonals.push_back({a, c});
			next_idx[a] = c;
			prev_idx[c] = a;
			--remaining;
			steps_since_clip = 0;
			current = c;
		} else {
			current = next_idx[current];
			++steps_since_clip;
			if (steps_since_clip > remaining + 1) {
				// No ear found after a full pass over the remaining vertices --
				// numerical degeneracy (near-duplicate points, zero-area
				// slivers). Bail out; caller falls back to a direct chord.
				return {diagonals, false};
			}
		}
	}
	return {diagonals, true};
}

struct AStarResult {
	std::vector<size_t> path;  // vertex indices, start..goal inclusive
	bool found;
};

// A* shortest path from `start` to `goal` over `graph` (adjacency list of
// (neighbor, edge weight) pairs), using Euclidean distance-to-goal as the
// heuristic -- admissible and consistent, since it never overestimates any
// path's remaining length through these same point coordinates.
inline AStarResult a_star_shortest_path(const std::vector<Point2D>& pts,
                                         const std::vector<std::vector<std::pair<size_t, double>>>& graph,
                                         size_t start, size_t goal) {
	size_t n = pts.size();
	std::vector<double> g_score(n, std::numeric_limits<double>::infinity());
	std::vector<size_t> came_from(n, (size_t)-1);
	std::vector<bool> closed(n, false);
	g_score[start] = 0.0;

	auto h = [&](size_t v) { return distance(pts[v], pts[goal]); };

	using QueueItem = std::pair<double, size_t>;
	std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> open;
	open.push({h(start), start});

	while (!open.empty()) {
		auto [f, u] = open.top();
		open.pop();
		(void)f;
		if (closed[u]) continue;
		if (u == goal) break;
		closed[u] = true;

		for (auto& [v, w] : graph[u]) {
			if (closed[v]) continue;
			double tentative = g_score[u] + w;
			if (tentative < g_score[v]) {
				g_score[v] = tentative;
				came_from[v] = u;
				open.push({tentative + h(v), v});
			}
		}
	}

	if (!(g_score[goal] < std::numeric_limits<double>::infinity())) {
		return {{}, false};
	}

	std::vector<size_t> path;
	for (size_t v = goal;; v = came_from[v]) {
		path.push_back(v);
		if (v == start) break;
	}
	std::reverse(path.begin(), path.end());
	return {std::move(path), true};
}

// The interior-respecting shortest-path cut between polygon vertices i and
// j, as a sequence of vertex indices [i, ..., j]. Falls back to the direct
// two-point chord {i, j} if triangulation fails AND the direct chord isn't
// valid either, or (defensively) if A* finds no path at all -- both should
// be rare and only arise from degenerate input geometry (in which case
// split_polygon_along_path below still produces a reasonable, if imperfect
// for a concave pair, result: exactly this algorithm's pre-A* behavior).
inline std::vector<size_t> find_interior_path(const std::vector<Point2D>& pts, size_t i, size_t j) {
	size_t n = pts.size();
	std::vector<std::vector<std::pair<size_t, double>>> graph(n);
	auto add_edge = [&](size_t a, size_t b) {
		double w = distance(pts[a], pts[b]);
		graph[a].push_back({b, w});
		graph[b].push_back({a, w});
	};

	for (size_t k = 0; k < n; ++k) {
		add_edge(k, (k + 1) % n);
	}

	EarClipResult tri = ear_clip_diagonals(pts);
	if (tri.ok) {
		for (auto& [a, b] : tri.diagonals) add_edge(a, b);
	}

	bool direct_valid = is_valid_diagonal(pts, i, j);
	if (direct_valid) {
		add_edge(i, j);  // offer it explicitly -- see header comment above
	}

	if (!tri.ok && !direct_valid) {
		return {i, j};
	}

	AStarResult result = a_star_shortest_path(pts, graph, i, j);
	if (!result.found) {
		return {i, j};
	}
	return result.path;
}

// --- splitting the polygon along a found path, by face-tracing -------------

inline double angle_at(const std::vector<Point2D>& pts, size_t from_v, size_t to_v) {
	return std::atan2(pts[to_v].y - pts[from_v].y, pts[to_v].x - pts[from_v].x);
}

// Given arrival at vertex `to` via the directed edge (from -> to), returns
// the neighbor of `to` that continues tracing the SAME face: the one whose
// direction is the smallest rotation strictly after the reverse bearing
// (to -> from), rotating in the direction given by `orientation_sign` (+1 =
// counterclockwise, -1 = clockwise). This is the standard rule for walking
// one face of a planar straight-line graph, generalized here to vertices
// with any degree (not just degree 2/4): `neighbors.at(to)` must list every
// vertex `to` is connected to, with no duplicate entries.
//
// `orientation_sign` MUST match the polygon's own winding (its sign of the
// shoelace formula): a hand-built CCW-wound polygon and a CW-wound one
// (e.g. concavehull()'s own output -- empirically confirmed to be wound the
// opposite way from a naive CCW test polygon) require opposite rotational
// conventions for this rule to trace the correct face. Getting this wrong
// doesn't crash -- it silently traces the WRONG face (typically the whole
// original boundary cycle for both "pieces"), which is far more dangerous
// than a crash, since it was only caught here by explicitly checking
// against a known-correct piece count on real (not hand-built) polygons.
inline size_t next_face_vertex(const std::vector<Point2D>& pts,
                                const std::unordered_map<size_t, std::vector<size_t>>& neighbors, size_t from,
                                size_t to, double orientation_sign) {
	double reverse_angle = angle_at(pts, to, from);
	const std::vector<size_t>& nbrs = neighbors.at(to);

	size_t best = from;  // degenerate fallback: degree-1 vertex (shouldn't occur for n >= 3)
	double best_delta = std::numeric_limits<double>::infinity();
	for (size_t nb : nbrs) {
		if (nb == from) continue;
		double a = angle_at(pts, to, nb);
		double delta = orientation_sign * (a - reverse_angle);
		while (delta <= 1e-12) delta += 2.0 * M_PI;
		if (delta < best_delta) {
			best_delta = delta;
			best = nb;
		}
	}
	return best;
}

// Traces one face of the planar graph given by `neighbors`, starting with
// the directed edge (start_from -> start_to), stopping once that exact
// directed edge would be traversed again (i.e. the face has closed). Returns
// the traced vertices as a flat [x0,y0,x1,y1,...] polygon (closing back to
// its own first vertex is implicit, matching this codebase's convention).
// Every directed edge the trace consumes is recorded in `used`, so a caller
// enumerating every face of the graph (see split_polygon_along_path) knows
// which starting edges are already accounted for.
inline std::vector<double> trace_face(const std::vector<Point2D>& pts,
                                       const std::unordered_map<size_t, std::vector<size_t>>& neighbors,
                                       size_t start_from, size_t start_to, double orientation_sign,
                                       std::set<std::pair<size_t, size_t>>& used) {
	std::vector<double> flat;
	size_t n = pts.size();
	size_t max_steps = 2 * (n + neighbors.size()) + 8;  // generous bound on this face's edge count

	size_t from = start_from, to = start_to;
	for (size_t step = 0; step < max_steps; ++step) {
		flat.push_back(pts[from].x);
		flat.push_back(pts[from].y);
		used.insert({from, to});
		size_t next = next_face_vertex(pts, neighbors, from, to, orientation_sign);
		if (to == start_from && next == start_to) break;  // about to retrace the starting edge: done
		from = to;
		to = next;
	}
	return flat;
}

// Splits `pts` into its resulting polygons by tracing every bounded face of
// the planar graph formed by `pts`'s own boundary cycle plus `path`'s edges
// (see header comment for why this, rather than naive arc+path
// concatenation, is the correct general construction). `path` must be
// exactly what find_interior_path returns: [i, ..., j] with i != j.
//
// This is NOT always exactly 2 pieces. Each genuine new diagonal the path
// adds (a hop that isn't already a boundary edge) increases the bounded face
// count by one (Euler's formula: V - E + F = 2, and each new edge inside an
// existing face splits it into two). A single straight chord [i, j] adds
// exactly one diagonal and so yields exactly 2 pieces, matching the pre-A*
// behavior -- but a bent path that hops through k >= 2 genuine diagonals
// (e.g. the path re-touches the polygon's own boundary at an intermediate
// vertex, bypassing a small pocket on each side of that vertex) yields
// 1 + k pieces: the one large piece on the "far" side of the whole bend,
// plus one small pocket piece per diagonal. Treating the result as always
// exactly 2 pieces and discarding anything beyond the first two silently
// drops those pocket pieces' area entirely -- directly observed as an area
// deficit (summed piece area < original polygon area) on a hand-built
// example where a 2-diagonal path cut off two separate concave pockets on
// either side of one shared bend vertex.
inline std::vector<std::vector<double>> split_polygon_along_path(const std::vector<Point2D>& pts,
                                                                   const std::vector<size_t>& path) {
	size_t n = pts.size();

	// The rotational convention next_face_vertex uses to pick "the next edge
	// continuing this face" only traces the correct face if it matches the
	// polygon's own winding -- a CCW-wound polygon needs the opposite
	// rotational rule from a CW-wound one. Compute the winding directly
	// (shoelace sign) rather than assuming one; this codebase's own
	// concavehull()/ChiOutline outputs and a hand-built CCW test polygon
	// were empirically found to be wound oppositely, so assuming either
	// convention unconditionally silently traces the wrong face for the other.
	double signed_area2 = 0.0;
	for (size_t k = 0; k < n; ++k) {
		size_t k2 = (k + 1) % n;
		signed_area2 += pts[k].x * pts[k2].y - pts[k2].x * pts[k].y;
	}
	// CCW (signed_area2 > 0) needs clockwise rotation (orientation_sign -1);
	// CW needs counterclockwise (orientation_sign +1).
	double orientation_sign = signed_area2 > 0.0 ? -1.0 : 1.0;

	std::unordered_map<size_t, std::vector<size_t>> neighbors;
	neighbors.reserve(n);
	for (size_t k = 0; k < n; ++k) {
		neighbors[k].push_back((k + 1) % n);
		neighbors[k].push_back((k + n - 1) % n);
	}
	std::vector<std::pair<size_t, size_t>> genuine_diagonals;
	for (size_t m = 0; m + 1 < path.size(); ++m) {
		size_t a = path[m], b = path[m + 1];
		bool already_boundary = (b == (a + 1) % n) || (a == (b + 1) % n);
		if (!already_boundary) {
			neighbors[a].push_back(b);
			neighbors[b].push_back(a);
			genuine_diagonals.push_back({a, b});
		}
	}

	// Enumerate every bounded face by trying every directed edge that could
	// possibly start one and skipping those already consumed by an earlier
	// trace. A directed ORIGINAL boundary edge in the polygon's own forward
	// order (k -> k+1) always starts a bounded face (that's exactly what
	// `orientation_sign` was computed to guarantee); its reverse direction
	// instead belongs to the unbounded outer face, so it's deliberately
	// excluded here -- it would otherwise trace the entire thing inside-out.
	// Every direction of every genuine new diagonal, by contrast, is always
	// interior and belongs to some bounded face, so both directions are
	// offered as candidates.
	std::set<std::pair<size_t, size_t>> used;
	std::vector<std::pair<size_t, size_t>> candidates;
	for (size_t k = 0; k < n; ++k) {
		candidates.push_back({k, (k + 1) % n});
	}
	for (auto& [a, b] : genuine_diagonals) {
		candidates.push_back({a, b});
		candidates.push_back({b, a});
	}

	std::vector<std::vector<double>> pieces;
	for (auto& [sf, st] : candidates) {
		if (used.count({sf, st})) continue;
		pieces.push_back(trace_face(pts, neighbors, sf, st, orientation_sign, used));
	}
	return pieces;
}

// --- Graph-accepting generalization (custom EdgeWeightFn, no injected ------
//     straight-line chord, no ear-clipping) --------------------------------
//
// Unlike the flat-polygon find_interior_path above, this overload is handed
// a Graph that is ALREADY the full interior mesh (e.g. from
// ChiOutline::compute_graph()/AlphaOutline::compute_graph()/
// concavehull_graph()) -- there is nothing left to triangulate, and
// crucially the cut must be constrained to travel only along the graph's
// own actual edges. Injecting a synthetic straight-line i-j shortcut (the
// way the flat-polygon overload does via is_valid_diagonal) would defeat
// the point of having a real mesh with real, possibly custom-weighted,
// edges to route through -- so this never adds one.
//
// Uses plain Dijkstra (zero heuristic), not A* with a Euclidean heuristic:
// A*'s Euclidean-distance-to-goal heuristic is only admissible/consistent
// when edge weights themselves are Euclidean distances. `weight_fn` is
// arbitrary (e.g. degree-sum, which has no necessary relationship to
// Euclidean distance at all), so a Euclidean heuristic could overestimate
// true remaining cost and silently break A*'s optimality guarantee. A zero
// heuristic keeps the same algorithm structure while staying correct for
// any non-negative weight_fn, at the cost of A*'s usual speed edge over
// Dijkstra -- an acceptable trade at this codebase's typical graph sizes.
inline std::vector<size_t> find_interior_path(const Graph& g, size_t i, size_t j,
                                                const EdgeWeightFn& weight_fn = euclidean_edge_weight) {
	size_t n = g.vertices.size();
	std::vector<std::vector<std::pair<size_t, size_t>>> adjacency(n);  // (neighbor, edge_idx)
	for (size_t e = 0; e < g.edges.size(); ++e) {
		auto& [a, b] = g.edges[e];
		adjacency[a].push_back({b, e});
		adjacency[b].push_back({a, e});
	}

	std::vector<double> dist(n, std::numeric_limits<double>::infinity());
	std::vector<size_t> came_from(n, (size_t)-1);
	std::vector<bool> closed(n, false);
	dist[i] = 0.0;
	using QueueItem = std::pair<double, size_t>;
	std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> open;
	open.push({0.0, i});

	while (!open.empty()) {
		auto [d, u] = open.top();
		open.pop();
		if (closed[u]) continue;
		if (u == j) break;
		closed[u] = true;
		for (auto& [v, e] : adjacency[u]) {
			if (closed[v]) continue;
			double w = weight_fn(g, e);
			double nd = d + w;
			if (nd < dist[v]) {
				dist[v] = nd;
				came_from[v] = u;
				open.push({nd, v});
			}
		}
	}

	if (!(dist[j] < std::numeric_limits<double>::infinity())) {
		return {i, j};  // defensive: no path at all (disconnected graph) -- same fallback spirit as the flat-polygon overload
	}
	std::vector<size_t> path;
	for (size_t v = j;; v = came_from[v]) {
		path.push_back(v);
		if (v == i) break;
	}
	std::reverse(path.begin(), path.end());
	return path;
}

// Index-returning counterpart of trace_face: identical face-walking logic,
// but accumulates visited vertex INDICES (into `pts`) instead of flat
// coordinates, since split_graph_along_path (below) needs indices to build
// each resulting piece's own sub-Graph (remapped vertices/edges, sliced
// vertex_data/edge_data) -- the flat-coordinate trace_face above is left
// untouched for the existing flat-polygon call site.
inline std::vector<size_t> trace_face_indices(const std::vector<Point2D>& pts,
                                                const std::unordered_map<size_t, std::vector<size_t>>& neighbors,
                                                size_t start_from, size_t start_to, double orientation_sign,
                                                std::set<std::pair<size_t, size_t>>& used) {
	std::vector<size_t> indices;
	size_t n = pts.size();
	size_t max_steps = 2 * (n + neighbors.size()) + 8;

	size_t from = start_from, to = start_to;
	for (size_t step = 0; step < max_steps; ++step) {
		indices.push_back(from);
		used.insert({from, to});
		size_t next = next_face_vertex(pts, neighbors, from, to, orientation_sign);
		if (to == start_from && next == start_to) break;
		from = to;
		to = next;
	}
	return indices;
}

// One resulting piece of a Graph split: its own sub-mesh (remapped vertex
// indices 0..k-1, with vertex_data/edge_data sliced to match if `g` had
// any) plus its own boundary cycle, as indices into THIS piece's own
// `graph.vertices` (not the original `g`'s) -- ready to feed straight back
// into find_bottleneck/find_interior_path for recursion.
struct GraphPiece {
	Graph graph;
	std::vector<size_t> boundary;
};

// Graph-accepting generalization of split_polygon_along_path: splits `g`
// into its resulting pieces by tracing every bounded face of `g`'s own
// edges (no boundary-cycle-plus-diagonals construction needed -- `g`
// already contains every edge that matters, see find_interior_path's own
// comment above) the path crosses, exactly mirroring that function's
// "1 + k pieces for k genuine diagonals" behavior. Each traced face gives
// one piece's own boundary; every OTHER vertex/edge of `g` is assigned to
// whichever piece's boundary it falls inside (point-in-polygon), since a
// piece here is a full sub-MESH, not just a boundary cycle.
inline std::vector<GraphPiece> split_graph_along_path(const Graph& g, const std::vector<size_t>& boundary,
                                                        const std::vector<size_t>& path) {
	size_t nb = boundary.size();

	double signed_area2 = 0.0;
	for (size_t k = 0; k < nb; ++k) {
		size_t k2 = (k + 1) % nb;
		signed_area2 += g.vertices[boundary[k]].x * g.vertices[boundary[k2]].y -
		                g.vertices[boundary[k2]].x * g.vertices[boundary[k]].y;
	}
	double orientation_sign = signed_area2 > 0.0 ? -1.0 : 1.0;

	std::unordered_map<size_t, size_t> pos_in_boundary;
	for (size_t k = 0; k < nb; ++k) pos_in_boundary[boundary[k]] = k;

	// The face-TRACING graph must be ONLY the boundary cycle plus the path's
	// own genuine diagonals -- NOT all of g's edges. `g` is typically a full
	// triangulated mesh (e.g. from ChiOutline::compute_graph()) with many
	// interior diagonals that have nothing to do with this particular cut;
	// including them here would shatter the result into roughly one piece
	// per original triangle instead of the handful of pieces this cut
	// actually produces (exactly the same reasoning the flat-polygon
	// split_polygon_along_path's own `neighbors` construction follows: it
	// starts from only the n-cycle boundary wraparound, not every
	// ear-clipping diagonal). Every OTHER edge/vertex of `g` is assigned to
	// whichever traced piece it geometrically falls inside, further below.
	std::unordered_map<size_t, std::vector<size_t>> neighbors;
	for (size_t k = 0; k < nb; ++k) {
		size_t a = boundary[k], b = boundary[(k + 1) % nb];
		neighbors[a].push_back(b);
		neighbors[b].push_back(a);
	}

	std::vector<std::pair<size_t, size_t>> genuine_diagonals;
	for (size_t m = 0; m + 1 < path.size(); ++m) {
		size_t a = path[m], b = path[m + 1];
		bool already_boundary = false;
		auto ia = pos_in_boundary.find(a), ib = pos_in_boundary.find(b);
		if (ia != pos_in_boundary.end() && ib != pos_in_boundary.end()) {
			already_boundary = (ib->second == (ia->second + 1) % nb) || (ia->second == (ib->second + 1) % nb);
		}
		if (!already_boundary) {
			neighbors[a].push_back(b);
			neighbors[b].push_back(a);
			genuine_diagonals.push_back({a, b});
		}
	}

	std::set<std::pair<size_t, size_t>> used;
	std::vector<std::pair<size_t, size_t>> candidates;
	for (size_t k = 0; k < nb; ++k) {
		candidates.push_back({boundary[k], boundary[(k + 1) % nb]});
	}
	for (auto& [a, b] : genuine_diagonals) {
		candidates.push_back({a, b});
		candidates.push_back({b, a});
	}

	std::vector<std::vector<size_t>> face_boundaries;
	for (auto& [sf, st] : candidates) {
		if (used.count({sf, st})) continue;
		face_boundaries.push_back(trace_face_indices(g.vertices, neighbors, sf, st, orientation_sign, used));
	}

	std::vector<GraphPiece> out;
	for (auto& face : face_boundaries) {
		std::unordered_set<size_t> included(face.begin(), face.end());
		for (size_t v = 0; v < g.vertices.size(); ++v) {
			if (included.count(v)) continue;
			if (point_in_polygon(g.vertices[v], [&] {
				    std::vector<Point2D> poly;
				    poly.reserve(face.size());
				    for (size_t idx : face) poly.push_back(g.vertices[idx]);
				    return poly;
			    }())) {
				included.insert(v);
			}
		}

		std::unordered_map<size_t, size_t> remap;
		remap.reserve(included.size());
		GraphPiece piece;
		piece.graph.vertices.reserve(included.size());
		bool has_vertex_data = !g.vertex_data.empty();
		if (has_vertex_data) piece.graph.vertex_data.reserve(included.size());
		for (size_t v : included) {
			remap[v] = piece.graph.vertices.size();
			piece.graph.vertices.push_back(g.vertices[v]);
			if (has_vertex_data) piece.graph.vertex_data.push_back(g.vertex_data[v]);
		}
		bool has_edge_data = !g.edge_data.empty();
		for (size_t e = 0; e < g.edges.size(); ++e) {
			auto& [a, b] = g.edges[e];
			if (!included.count(a) || !included.count(b)) continue;
			piece.graph.edges.push_back({remap[a], remap[b]});
			if (has_edge_data) piece.graph.edge_data.push_back(g.edge_data[e]);
		}
		piece.boundary.reserve(face.size());
		for (size_t idx : face) piece.boundary.push_back(remap[idx]);

		out.push_back(std::move(piece));
	}
	return out;
}
