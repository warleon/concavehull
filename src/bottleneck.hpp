#pragma once

// Bottleneck detection and cutting (steps 2-3 of):
//
//   Wang, H., Zhang, H., & Ray, N. (2012). Clump splitting via bottleneck
//   detection and shape classification. Pattern Recognition, 45(7), 2780-2787.
//
// Given a simple polygon (e.g. the boundary returned by concavehull() or one
// polygon from ChiOutline::compute()), BottleneckSplit repeatedly:
//   1. asks a pluggable SplitCriterion (see split_criterion.hpp) whether the
//      polygon still looks like a clump of multiple merged objects,
//   2. if so, finds the pair of boundary vertices (A*, B*) minimizing the
//      bottleneck cost
//          Es(A,B) = dist(A,B) / min(arc(A,B), arc(B,A))                (Eq. 1-2)
//      where arc(A,B)/arc(B,A) are the two boundary-path lengths between A
//      and B, and
//   3. cuts the polygon into two simple sub-polygons along the shortest
//      interior-respecting path between A* and B* (see interior_path.hpp).
//
// Step 3 in the paper finds the optimal cut via a weighted shortest path
// over image-intensity edge strength (Eqs. 3-4), using Dijkstra's algorithm.
// We only have polygon geometry here, no source image, so there is no
// intensity information to route the cut around -- interior_path.hpp instead
// treats the polygon's own triangulated interior as the thing being routed
// through, and finds the shortest such path via A* (edge weights = Euclidean
// edge length). See interior_path.hpp's own header comment for the full
// reasoning, including why this reduces to exactly the direct straight-line
// chord whenever that chord is already a valid cut.
//
// Recursion continues on each resulting piece until the criterion says stop,
// a piece gets too small to split further, or `max_depth` is reached.

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "interior_path.hpp"
#include "split_criterion.hpp"

class BottleneckSplit {
public:
	// `min_piece_area` rejects any candidate split that would produce a piece
	// smaller than this (absolute, same units as the input polygon's
	// coordinates -- e.g. pixels^2 for a mask-derived polygon). Default 0.0
	// disables the check entirely, preserving old behavior. This matters
	// because `should_split`'s geometric criteria can't distinguish a real
	// neck between two touching objects from the small non-convex notches a
	// single object's own *organic* boundary naturally has (e.g. individual
	// berries bulging within one mask-derived grape-bunch contour) -- left
	// unchecked, recursion keeps chipping near-zero-area slivers off such a
	// boundary indefinitely (observed directly: dozens of sub-1%-area
	// fragments from a single lobed input polygon). A non-zero floor turns
	// "this split would produce a sliver" into the same kind of rejection as
	// the existing no-diagonal/no-bottleneck-pair fallbacks below, rather
	// than a hard vertex-count/depth cutoff that a lobed-but-small polygon
	// can still slip past.
	explicit BottleneckSplit(std::shared_ptr<SplitCriterion> criterion = nullptr, double min_arc_fraction = 0.1,
	                          int max_depth = 12, double min_piece_area = 0.0)
		: criterion_(criterion ? std::move(criterion) : std::make_shared<BottleneckCostCriterion>()),
		  min_arc_fraction_(min_arc_fraction),
		  max_depth_(max_depth),
		  min_piece_area_(min_piece_area) {
		if (min_arc_fraction_ <= 0.0 || min_arc_fraction_ >= 0.5) {
			throw std::invalid_argument("min_arc_fraction must be in (0, 0.5)");
		}
		if (max_depth_ < 0) {
			throw std::invalid_argument("max_depth must be non-negative");
		}
		if (min_piece_area_ < 0.0) {
			throw std::invalid_argument("min_piece_area must be non-negative");
		}
	}

	// Recursively splits one simple polygon (flat [x0,y0,x1,y1,...]) into a
	// list of sub-polygons using the same flat convention.
	std::vector<std::vector<double>> split(const std::vector<double>& polygon) const {
		if (polygon.size() % 2 != 0) {
			throw std::invalid_argument("polygon coordinates must come in (x, y) pairs");
		}
		std::vector<Point2D> pts = to_points(polygon);
		if (pts.size() < 3) {
			throw std::invalid_argument("polygon must have at least 3 vertices");
		}
		std::vector<std::vector<double>> out;
		split_recursive(std::move(pts), 0, out);
		return out;
	}

	// Convenience: run split() over every polygon in a collection (e.g. the
	// output of ChiOutline::compute()) and flatten the results.
	std::vector<std::vector<double>> split_all(const std::vector<std::vector<double>>& polygons) const {
		std::vector<std::vector<double>> out;
		for (auto& poly : polygons) {
			auto parts = split(poly);
			out.insert(out.end(), std::make_move_iterator(parts.begin()), std::make_move_iterator(parts.end()));
		}
		return out;
	}

private:
	std::shared_ptr<SplitCriterion> criterion_;
	double min_arc_fraction_;
	int max_depth_;
	double min_piece_area_;

	static std::vector<Point2D> to_points(const std::vector<double>& flat) {
		std::vector<Point2D> pts;
		pts.reserve(flat.size() / 2);
		for (size_t i = 0; i + 1 < flat.size(); i += 2) {
			pts.push_back({flat[i], flat[i + 1]});
		}
		return pts;
	}

	static std::vector<double> to_flat(const std::vector<Point2D>& pts) {
		std::vector<double> flat;
		flat.reserve(pts.size() * 2);
		for (auto& p : pts) {
			flat.push_back(p.x);
			flat.push_back(p.y);
		}
		return flat;
	}

	void split_recursive(std::vector<Point2D> pts, int depth, std::vector<std::vector<double>>& out) const {
		bool leaf = pts.size() < 6 || depth >= max_depth_ || !criterion_->should_split(pts);
		if (leaf) {
			out.push_back(to_flat(pts));
			return;
		}

		// Eq. 1-2: find the pair of vertex indices minimizing the bottleneck
		// cost, restricted to pairs where both resulting arcs are at least
		// `min_arc_fraction` of the perimeter (and both resulting pieces
		// would have >= 3 vertices).
		BottleneckResult r = find_bottleneck(pts, min_arc_fraction_);
		size_t i = r.i, j = r.j;
		if (i == SIZE_MAX) {
			out.push_back(to_flat(pts));  // criterion wants a split, geometry can't give one
			return;
		}

		// Step 3 (Eqs. 3-4): the shortest interior-respecting path from i to j
		// -- [i, ..., j] as vertex indices into `pts`. Reduces to exactly {i, j}
		// (no intermediate vertices) whenever the direct chord is itself a
		// valid cut. Then split along it by face-tracing (NOT naive arc+path
		// concatenation -- see interior_path.hpp's header comment for why
		// that's the one correct general construction).
		std::vector<size_t> cut_path = find_interior_path(pts, i, j);

		// Degenerate case: if every consecutive pair in cut_path is already an
		// original boundary edge (no genuine diagonal anywhere along it), the
		// path contributes NO new edge to the polygon's graph at all -- it's
		// just retracing one of the two existing boundary arcs. There is then
		// no actual interior cut between i and j for this pair (this (i, j)
		// pair doesn't correspond to a real separable waist), and
		// split_polygon_along_path would (correctly, given its input) return
		// two copies of the entire unmodified polygon, which would otherwise
		// recurse without ever making progress. Treat exactly like the
		// existing "geometry can't give a split" fallback above.
		size_t n = pts.size();
		bool any_diagonal = false;
		for (size_t m = 0; m + 1 < cut_path.size(); ++m) {
			size_t a = cut_path[m], b = cut_path[m + 1];
			if (b != (a + 1) % n && a != (b + 1) % n) {
				any_diagonal = true;
				break;
			}
		}
		if (!any_diagonal) {
			out.push_back(to_flat(pts));
			return;
		}

		std::vector<std::vector<double>> pieces = split_polygon_along_path(pts, cut_path);

		if (min_piece_area_ > 0.0) {
			for (auto& piece_flat : pieces) {
				if (polygon_area(to_points(piece_flat)) < min_piece_area_) {
					out.push_back(to_flat(pts));  // this cut would create a degenerate sliver -- reject it
					return;
				}
			}
		}

		for (auto& piece_flat : pieces) {
			split_recursive(to_points(piece_flat), depth + 1, out);
		}
	}
};
