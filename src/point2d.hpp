#pragma once

// Extracted from split_criterion.hpp so that both split_criterion.hpp and
// graph.hpp can depend on Point2D without a circular include between them
// (split_criterion.hpp's Graph-accepting find_bottleneck needs graph.hpp;
// graph.hpp needs Point2D). No other dependencies.

#include <cmath>

struct Point2D {
	double x, y;
};

inline double distance(const Point2D& a, const Point2D& b) {
	double dx = a.x - b.x;
	double dy = a.y - b.y;
	return std::sqrt(dx * dx + dy * dy);
}
