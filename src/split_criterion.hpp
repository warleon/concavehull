#pragma once

// Shared 2D polygon utilities, plus a pluggable "should this polygon be
// split?" decision -- the shape-classification step of:
//
//   Wang, H., Zhang, H., & Ray, N. (2012). Clump splitting via bottleneck
//   detection and shape classification. Pattern Recognition, 45(7), 2780-2787.
//
// That paper trains an application-specific SVM over shape features offline;
// we don't have labelled training data, so instead this header exposes the
// decision as a small polymorphic interface (SplitCriterion) that can be
// swapped out independently of the bottleneck-cutting mechanics in
// bottleneck.hpp. Two concrete criteria are provided below.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

struct Point2D {
	double x, y;
};

inline double distance(const Point2D& a, const Point2D& b) {
	double dx = a.x - b.x;
	double dy = a.y - b.y;
	return std::sqrt(dx * dx + dy * dy);
}

// Shoelace formula. Works for any simple polygon regardless of winding.
inline double polygon_area(const std::vector<Point2D>& pts) {
	double a = 0.0;
	size_t n = pts.size();
	for (size_t i = 0; i < n; ++i) {
		size_t j = (i + 1) % n;
		a += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
	}
	return std::abs(a) * 0.5;
}

// Andrew's monotone chain convex hull, O(n log n).
inline std::vector<Point2D> convex_hull(std::vector<Point2D> pts) {
	std::sort(pts.begin(), pts.end(), [](const Point2D& a, const Point2D& b) {
		return a.x < b.x || (a.x == b.x && a.y < b.y);
	});
	pts.erase(std::unique(pts.begin(), pts.end(),
	                       [](const Point2D& a, const Point2D& b) { return a.x == b.x && a.y == b.y; }),
	          pts.end());

	size_t n = pts.size();
	if (n < 3) return pts;

	auto cross = [](const Point2D& o, const Point2D& a, const Point2D& b) {
		return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
	};

	std::vector<Point2D> hull(2 * n);
	size_t k = 0;
	for (size_t i = 0; i < n; ++i) {
		while (k >= 2 && cross(hull[k - 2], hull[k - 1], pts[i]) <= 0) --k;
		hull[k++] = pts[i];
	}
	size_t lower = k + 1;
	for (long i = (long)n - 2; i >= 0; --i) {
		while (k >= lower && cross(hull[k - 2], hull[k - 1], pts[(size_t)i]) <= 0) --k;
		hull[k++] = pts[(size_t)i];
	}
	hull.resize(k - 1);
	return hull;
}

// area(polygon) / area(convex_hull(polygon)), in (0, 1]. 1.0 means the
// polygon already equals its own convex hull.
inline double solidity(const std::vector<Point2D>& pts) {
	std::vector<Point2D> hull = convex_hull(pts);
	if (hull.size() < 3) return 1.0;
	double hull_area = polygon_area(hull);
	if (hull_area <= 0.0) return 1.0;
	return polygon_area(pts) / hull_area;
}

// The core bottleneck-detection quantity (Wang, Zhang & Ray 2012, Eq. 1-2):
// the pair of vertex indices i < j minimizing
//     dist(pts[i], pts[j]) / min(arc(i, j), arc(j, i))
// restricted to pairs where both arcs are >= min_arc_fraction * perimeter
// and both resulting pieces would have >= 3 vertices. `i` is SIZE_MAX (cost
// +inf) if no pair qualifies. Shared by BottleneckCostCriterion below and by
// BottleneckSplit's own cutting step in bottleneck.hpp, so the "should I
// split" decision and the "where do I cut" decision can agree by construction.
struct BottleneckResult {
	double cost;
	size_t i, j;
};

inline BottleneckResult find_bottleneck(const std::vector<Point2D>& pts, double min_arc_fraction) {
	size_t n = pts.size();
	std::vector<double> cum(n, 0.0);
	for (size_t k = 1; k < n; ++k) {
		cum[k] = cum[k - 1] + distance(pts[k - 1], pts[k]);
	}
	double perimeter = cum[n - 1] + distance(pts[n - 1], pts[0]);
	double min_arc_len = min_arc_fraction * perimeter;

	BottleneckResult best{std::numeric_limits<double>::infinity(), SIZE_MAX, SIZE_MAX};
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = i + 1; j < n; ++j) {
			double arc_ij = cum[j] - cum[i];
			double arc_ji = perimeter - arc_ij;
			double min_arc = std::min(arc_ij, arc_ji);
			if (min_arc < min_arc_len) continue;

			size_t piece1 = j - i + 1;
			size_t piece2 = n - piece1 + 2;
			if (piece1 < 3 || piece2 < 3) continue;

			double cost = distance(pts[i], pts[j]) / min_arc;
			if (cost < best.cost) best = {cost, i, j};
		}
	}
	return best;
}

class SplitCriterion {
public:
	virtual ~SplitCriterion() = default;
	virtual bool should_split(const std::vector<Point2D>& pts) const = 0;
};

// Direct implementation of the chi-shape paper's L2 error norm (Duckham et
// al. 2008, Section 6.1), applied here between a polygon and its own convex
// hull rather than between a chi-shape and a ground-truth shape:
//
//   L2 = area(symmetric_difference(hull, polygon)) / area(polygon)
//
// Since a polygon is always contained in its own convex hull, polygon - hull
// is empty, so this reduces to (area(hull) - area(polygon)) / area(polygon)
// = 1/solidity - 1.
//
// CAUTION: unlike the paper's original use (comparing two shapes of similar
// size), this ratio is NOT bounded to [0, 1]. It is 0 when the polygon is
// already convex and grows without bound as the polygon's area shrinks
// relative to its hull (a thin "peanut" shape with solidity 0.1 already
// gives L2 = 9). Pick `l2_threshold` accordingly -- it is not a probability
// or a fraction. The default of 0.18 corresponds to solidity ~= 0.85, the
// same cutoff ConvexDeficitCriterion's default represents. If you want a
// threshold that *is* naturally in [0, 1], use ConvexDeficitCriterion below
// instead.
//
// IMPORTANT, for both criteria below: a polygon's solidity is sensitive to
// how it was generated, not just to genuine clump merging. A chi-shape/
// chi-outline polygon built with a low chi/chi_factor is, by construction,
// already quite non-convex from ordinary boundary noise alone -- a single,
// perfectly good, unmerged blob can easily show solidity in the 0.55-0.85
// range at low chi, which overlaps the range a genuinely merged clump would
// show. Conversely, heavily overlapping objects can produce a union whose
// outline is nearly convex, hiding the bottleneck entirely. There is no
// threshold that is robust to both failure modes at once. For best results,
// feed BottleneckSplit a moderate-to-high chi/chi_factor (a smoother
// boundary, e.g. 0.3-0.6) so that remaining concavities are more likely to
// reflect real necks, and calibrate the threshold on a few representative
// examples from your own data -- this manual calibration stands in for the
// trained SVM the original paper uses for exactly this decision.
class ConvexHullL2Criterion : public SplitCriterion {
public:
	explicit ConvexHullL2Criterion(double l2_threshold = 0.18) : l2_threshold_(l2_threshold) {
		if (l2_threshold_ < 0.0) {
			throw std::invalid_argument("l2_threshold must be non-negative");
		}
	}

	bool should_split(const std::vector<Point2D>& pts) const override {
		double s = solidity(pts);
		if (s <= 0.0) return false;  // degenerate polygon; nothing sensible to do
		double l2 = 1.0 / s - 1.0;
		return l2 > l2_threshold_;
	}

private:
	double l2_threshold_;
};

// Recommended alternative when a [0, 1)-bounded threshold is wanted: the
// same symmetric-difference area, but normalized by the hull's area instead
// of the polygon's own area:
//
//   deficit = (area(hull) - area(polygon)) / area(hull) = 1 - solidity
//
// Because a polygon's area can never exceed its own convex hull's area,
// `deficit` is always in [0, 1): 0 for an already-convex polygon, approaching
// 1 only as the polygon's area vanishes relative to its hull.
//
// Other criteria that are naturally bounded to [0, 1] if you want options:
//   - solidity() itself (threshold directly on area(polygon)/area(hull),
//     splitting when it drops *below* the threshold rather than above);
//   - convexity = hull_perimeter / polygon_perimeter, in (0, 1] for simple
//     polygons;
//   - a squashed version of the raw L2 above, e.g. L2 / (1 + L2), which maps
//     [0, inf) to [0, 1) monotonically if you specifically want to keep the
//     paper's exact L2 numerator/denominator convention but still have a
//     bounded knob to tune.
//
// Caution from empirical testing: this and ConvexHullL2Criterion both judge
// the polygon *as a whole*, via total area lost to concavity. That lumps a
// single genuine neck together with many small, unrelated boundary wiggles
// (which a chi-shape/chi-outline polygon has plenty of at low chi, even for
// a single unmerged blob) into one number. In practice this makes both
// criteria prone to both false positives (noisy single blobs) and false
// negatives (heavily overlapping multi-blob unions that read as "mostly
// convex" overall) depending on chi and the threshold. BottleneckCostCriterion
// below, which judges only the single best candidate neck rather than the
// polygon's overall shape, was found to discriminate far more reliably in
// practice and is BottleneckSplit's default.
class ConvexDeficitCriterion : public SplitCriterion {
public:
	explicit ConvexDeficitCriterion(double deficit_threshold = 0.15) : deficit_threshold_(deficit_threshold) {
		if (deficit_threshold_ < 0.0 || deficit_threshold_ >= 1.0) {
			throw std::invalid_argument("deficit_threshold must be in [0, 1)");
		}
	}

	bool should_split(const std::vector<Point2D>& pts) const override {
		double deficit = 1.0 - solidity(pts);
		return deficit > deficit_threshold_;
	}

private:
	double deficit_threshold_;
};

// Recommended default: uses the bottleneck cost itself (Eq. 1-2) as the
// classification signal, rather than a global convexity measure over the
// whole polygon. Splits only if some pair of boundary points has a cost
//     Es(A, B) = dist(A, B) / min(arc(A, B), arc(B, A))
// below `cost_threshold`. This is a *local* signal: it only reacts to one
// genuinely narrow, deep neck, and is far less sensitive to generic small-
// scale boundary noise spread around the rest of the shape (e.g. from a low
// chi/chi_factor) than the convex-hull-area criteria above. It is exactly
// the quantity BottleneckSplit's own cutting step already computes, just
// exposed as a standalone, tunable decision.
//
// For reference, a pair of *adjacent* polygon vertices always has cost
// exactly 1.0 (dist/arc = edge_length / edge_length), so 1.0 is a loose
// upper bound for "no real neck here".
//
// Empirically (tested on Gaussian-blob point clouds of various sizes, with
// polygons from concavehull()/chi-shape at various chi_factor): the dominant
// factor in how separable "genuinely touching objects" are from "one noisy
// blob" is the chi/chi_factor used to build the polygon, far more than point
// count. At low chi_factor (~0.2-0.3) chi-shape intentionally carves many
// small non-convex notches into *any* point cloud, single blobs included, so
// their best candidate cost lands in the same range a real neck would (no
// threshold separates them reliably -- true-positive and true-negative rates
// both land around 40-60%). At a moderate-to-high chi_factor (~0.5-0.7) most
// of that small-scale noise is smoothed away while a genuine wide/deep neck
// remains visible, giving a true separation in the cost: in that regime a
// threshold of 0.45 scored roughly 95% correct rejection of single unmerged
// blobs and roughly 80% correct detection of genuinely touching ones across
// 30 random trials at chi_factor=0.6. That is the default here -- but it
// comes with the expectation that the polygon you hand to BottleneckSplit
// was itself built with chi/chi_factor around 0.5 or higher. At lower
// chi/chi_factor, expect this (and ConvexDeficitCriterion/ConvexHullL2Criterion
// above) to be unreliable regardless of threshold -- this is a real limitation,
// not a tuning problem, and matches the fact that the original paper's method
// was designed for smooth binary-mask contours from real images, not the
// comparatively jagged boundaries sparse point-cloud triangulations produce
// at low chi.
class BottleneckCostCriterion : public SplitCriterion {
public:
	explicit BottleneckCostCriterion(double cost_threshold = 0.45, double min_arc_fraction = 0.1)
		: cost_threshold_(cost_threshold), min_arc_fraction_(min_arc_fraction) {
		if (cost_threshold_ < 0.0) {
			throw std::invalid_argument("cost_threshold must be non-negative");
		}
		if (min_arc_fraction_ <= 0.0 || min_arc_fraction_ >= 0.5) {
			throw std::invalid_argument("min_arc_fraction must be in (0, 0.5)");
		}
	}

	bool should_split(const std::vector<Point2D>& pts) const override {
		BottleneckResult r = find_bottleneck(pts, min_arc_fraction_);
		return r.i != SIZE_MAX && r.cost < cost_threshold_;
	}

private:
	double cost_threshold_;
	double min_arc_fraction_;
};
