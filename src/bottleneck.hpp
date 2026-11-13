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
//   3. cuts the polygon into two simple sub-polygons along the straight
//      chord A*-B*.
//
// Step 3 in the paper finds the optimal cut via a weighted shortest path
// over image-intensity edge strength (Eqs. 3-4), using Dijkstra's algorithm.
// We only have polygon geometry here, no source image, so there is no
// intensity information to route around -- which is exactly the case the
// paper itself says the weighted shortest path "degenerates" to: the
// straight-line cut between A* and B*.
//
// Recursion continues on each resulting piece until the criterion says stop,
// a piece gets too small to split further, or `max_depth` is reached.

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "split_criterion.hpp"

class BottleneckSplit {
public:
	explicit BottleneckSplit(std::shared_ptr<SplitCriterion> criterion = nullptr, double min_arc_fraction = 0.1,
	                          int max_depth = 12)
		: criterion_(criterion ? std::move(criterion) : std::make_shared<BottleneckCostCriterion>()),
		  min_arc_fraction_(min_arc_fraction),
		  max_depth_(max_depth) {
		if (min_arc_fraction_ <= 0.0 || min_arc_fraction_ >= 0.5) {
			throw std::invalid_argument("min_arc_fraction must be in (0, 0.5)");
		}
		if (max_depth_ < 0) {
			throw std::invalid_argument("max_depth must be non-negative");
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

		std::vector<Point2D> piece1(pts.begin() + (long)i, pts.begin() + (long)j + 1);

		std::vector<Point2D> piece2;
		piece2.reserve(pts.size() - piece1.size() + 2);
		for (size_t k = j; k < pts.size(); ++k) piece2.push_back(pts[k]);
		for (size_t k = 0; k <= i; ++k) piece2.push_back(pts[k]);

		split_recursive(std::move(piece1), depth + 1, out);
		split_recursive(std::move(piece2), depth + 1, out);
	}
};
