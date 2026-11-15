#pragma once

// Stoer & Wagner (1997) global minimum cut algorithm for a weighted,
// undirected, non-negative-weight graph: the minimum-weight set of edges
// whose removal splits the graph into exactly two non-empty parts, found
// without designating any source/sink pair (unlike a max-flow/min s-t-cut,
// which requires one, and would need to be run over all C(n,2) pairs --
// or just n-1 fixed-source pairs by the standard reduction -- to get the
// same global answer). Classic "minimum cut phase" formulation, O(V^3) with
// a dense adjacency-matrix representation -- appropriate at the object
// -graph scale this is used for (GraphBottleneckSplit, in
// graph_bottleneck_split.hpp): splitting one cluster of a few tens to low
// hundreds of already-detected objects, not a large global graph.
//
// Reference: M. Stoer, F. Wagner (1997). "A Simple Min-Cut Algorithm."
// Journal of the ACM, 44(4), 585-591.

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

struct MinCutResult {
	double weight;
	// One side of the cut, as indices into the original 0..n-1 vertex
	// numbering; the other side is every vertex not listed here. Which side
	// is reported is arbitrary (Stoer-Wagner finds an unordered bipartition).
	std::vector<size_t> side;
};

// `weights` is a dense n x n symmetric adjacency-weight matrix
// (weights[i][j] == weights[j][i], diagonal ignored, 0 meaning "no edge").
// Requires n >= 2. If the graph is disconnected, correctly returns a cut of
// weight 0 separating two of its components (a global min cut of a
// disconnected graph is trivially 0) -- callers that need a *merge*
// decision, not just the numeric cut value, should only call this on an
// already-connected cluster.
inline MinCutResult stoer_wagner_min_cut(std::vector<std::vector<double>> weights) {
	size_t n = weights.size();
	if (n < 2) {
		throw std::invalid_argument("stoer_wagner_min_cut requires at least 2 vertices");
	}

	std::vector<bool> active(n, true);
	std::vector<std::vector<size_t>> members(n);
	for (size_t i = 0; i < n; ++i) members[i] = {i};

	double best_weight = std::numeric_limits<double>::infinity();
	std::vector<size_t> best_side;

	size_t remaining = n;
	while (remaining > 1) {
		// --- one "minimum cut phase": grow A by always adding the active,
		// not-yet-included vertex most tightly connected to A so far ---
		std::vector<bool> in_a(n, false);
		std::vector<double> w(n, 0.0);

		size_t first_active = (size_t)-1;
		for (size_t i = 0; i < n; ++i) {
			if (active[i]) {
				first_active = i;
				break;
			}
		}
		in_a[first_active] = true;
		for (size_t i = 0; i < n; ++i) {
			if (active[i] && i != first_active) w[i] = weights[first_active][i];
		}

		size_t prev = first_active, last = first_active;
		for (size_t step = 1; step < remaining; ++step) {
			size_t z = (size_t)-1;
			double best_w = -1.0;
			for (size_t i = 0; i < n; ++i) {
				if (active[i] && !in_a[i] && w[i] > best_w) {
					best_w = w[i];
					z = i;
				}
			}
			in_a[z] = true;
			prev = last;
			last = z;
			for (size_t i = 0; i < n; ++i) {
				if (active[i] && !in_a[i]) w[i] += weights[z][i];
			}
		}

		// "cut-of-the-phase": separates {everything folded into `last`} from
		// the rest of the current (super-vertex) graph.
		double cut_weight = w[last];
		if (cut_weight < best_weight) {
			best_weight = cut_weight;
			best_side = members[last];
		}

		// merge `last` into `prev` (Stoer-Wagner's key step: the global min
		// cut either IS the cut-of-the-phase just found, or doesn't separate
		// this particular pair -- so it's always safe to merge them).
		for (size_t i = 0; i < n; ++i) {
			if (active[i] && i != prev && i != last) {
				weights[prev][i] += weights[last][i];
				weights[i][prev] += weights[i][last];
			}
		}
		members[prev].insert(members[prev].end(), members[last].begin(), members[last].end());
		active[last] = false;
		--remaining;
	}

	return {best_weight, std::move(best_side)};
}
