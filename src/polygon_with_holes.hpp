#pragma once

#include <vector>

// A polygon that may have interior holes: one exterior ring plus zero or
// more hole rings, each a flat [x0,y0,x1,y1,...] coordinate vector (same
// convention as every other polygon in this codebase -- concavehull(),
// ChiOutline::compute(), BottleneckSplit::split()). Produced by AlphaOutline
// (see alpha_outline.hpp), which -- unlike ChiOutline/concavehull -- does
// not force a single simple (hole-free) polygon per connected region.
struct PolygonWithHoles {
	std::vector<double> exterior;
	std::vector<std::vector<double>> holes;
};
