#pragma once

// GraphBottleneckSplit: the graph-level generalization of BottleneckSplit
// (bottleneck.hpp) for a cluster of already-detected, individuated objects
// (e.g. berries from an ML model) represented as an ObjectGraph
// (object_cluster.hpp), rather than a polygon boundary.
//
// BottleneckCostCriterion's cost, Es(A,B) = dist(A,B) / min(arc(A,B),
// arc(B,A)), has no meaning for a discrete object graph -- there is no
// "arc" between two objects, only a graph of pairwise distances. The direct
// graph-theoretic analogue of "find the narrowest connection and cut there"
// is a minimum cut (stoer_wagner.hpp) separating the graph into two parts,
// found without designating any source/sink pair. Minimizing a SUM of edge
// weights requires those weights to represent connection STRENGTH, not
// distance (a min cut over raw distances would instead find the single
// pair of objects that happen to be *farthest* apart and call that the
// weakest link, which is backwards) -- so every distance here is converted
// to an affinity weight 1/(distance + eps) before being handed to the min
// -cut solver, mirroring the standard affinity-weighting used in
// spectral/graph-based clustering.
//
// Specifically uses best_ratio_cut, not the plain global min cut: a plain
// min cut is well known to prefer isolating a tiny, weakly-connected corner
// of the graph over finding a genuinely balanced separating cut -- verified
// directly on a two-equal-size-cluster test case, where stoer_wagner_min_cut
// reported a 3-vs-21 split (lower raw weight) while best_ratio_cut correctly
// found the true 12-vs-12 split sitting in an earlier phase. See
// best_ratio_cut's own comment in stoer_wagner.hpp for the full reasoning.
//
// Mirrors BottleneckSplit's own recursive shape and SplitCriterion pattern
// exactly (should-I-split, then cut, then recurse on each resulting piece),
// but every "piece" here is a list of object indices, not polygon geometry
// -- no triangulation-with-holes is needed anywhere in this file, since it
// never touches polygon boundaries, only the object graph already built by
// ObjectClusterGraph.

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "object_cluster.hpp"
#include "stoer_wagner.hpp"

// Converts a distance-weighted induced subgraph (restricted to `group`'s own
// objects) into an affinity-weighted dense matrix and finds its global min
// cut via Stoer-Wagner. Returns the cut weight (affinity units) and a
// GLOBAL-object-id partition of `group` into two non-empty parts.
struct GraphMinCut {
	double weight;
	std::vector<size_t> side_a;
	std::vector<size_t> side_b;
};

inline GraphMinCut graph_min_cut(const std::vector<size_t>& group, const ObjectGraph& graph) {
	size_t k = group.size();
	if (k < 2) {
		throw std::invalid_argument("graph_min_cut requires at least 2 objects");
	}

	std::unordered_map<size_t, size_t> local_index;
	local_index.reserve(k);
	for (size_t i = 0; i < k; ++i) local_index[group[i]] = i;

	constexpr double kEps = 1e-9;
	std::vector<std::vector<double>> affinity(k, std::vector<double>(k, 0.0));
	for (size_t i = 0; i < k; ++i) {
		size_t global_i = group[i];
		for (auto& edge : graph.adjacency[global_i]) {
			auto it = local_index.find(edge.first);
			if (it == local_index.end()) continue;  // neighbor outside this group
			size_t j = it->second;
			double w = 1.0 / (edge.second + kEps);
			affinity[i][j] = w;
			affinity[j][i] = w;
		}
	}

	MinCutResult r = best_ratio_cut(affinity);

	std::vector<bool> in_side_a(k, false);
	for (size_t local : r.side) in_side_a[local] = true;

	GraphMinCut out;
	out.weight = r.weight;
	for (size_t i = 0; i < k; ++i) {
		(in_side_a[i] ? out.side_a : out.side_b).push_back(group[i]);
	}
	return out;
}

class GraphSplitCriterion {
public:
	virtual ~GraphSplitCriterion() = default;
	virtual bool should_split(const std::vector<size_t>& group, const ObjectGraph& graph) const = 0;
};

// Default criterion: cost = min_cut_weight / min(|A|, |B|), the direct graph
// analogue of BottleneckCostCriterion's arc-fraction normalization (there,
// cost is normalized by the shorter boundary arc; here, by the smaller
// resulting group's object count). should_split iff cost < cost_threshold
// AND both resulting sides have at least min_group_fraction * |group|
// objects (the graph analogue of min_arc_fraction).
class GraphBottleneckCostCriterion : public GraphSplitCriterion {
public:
	explicit GraphBottleneckCostCriterion(double cost_threshold = 0.45, double min_group_fraction = 0.1)
		: cost_threshold_(cost_threshold), min_group_fraction_(min_group_fraction) {
		if (cost_threshold_ < 0.0) {
			throw std::invalid_argument("cost_threshold must be non-negative");
		}
		if (min_group_fraction_ <= 0.0 || min_group_fraction_ >= 0.5) {
			throw std::invalid_argument("min_group_fraction must be in (0, 0.5)");
		}
	}

	bool should_split(const std::vector<size_t>& group, const ObjectGraph& graph) const override {
		if (group.size() < 2) return false;
		GraphMinCut cut = graph_min_cut(group, graph);
		size_t min_side = std::min(cut.side_a.size(), cut.side_b.size());
		size_t required = (size_t)std::ceil(min_group_fraction_ * (double)group.size());
		if (min_side < std::max<size_t>(required, 1)) return false;
		double cost = cut.weight / (double)min_side;
		return cost < cost_threshold_;
	}

private:
	double cost_threshold_;
	double min_group_fraction_;
};

class GraphBottleneckSplit {
public:
	explicit GraphBottleneckSplit(std::shared_ptr<GraphSplitCriterion> criterion = nullptr, int max_depth = 12)
		: criterion_(criterion ? std::move(criterion) : std::make_shared<GraphBottleneckCostCriterion>()),
		  max_depth_(max_depth) {
		if (max_depth_ < 0) {
			throw std::invalid_argument("max_depth must be non-negative");
		}
	}

	// Recursively splits `group` (a list of object indices into `graph`) into
	// final sub-groups -- no geometry involved, purely a partition of object
	// indices. Each returned group covers a disjoint subset of `group`, and
	// their union is exactly `group`.
	std::vector<std::vector<size_t>> split(const std::vector<size_t>& group, const ObjectGraph& graph) const {
		std::vector<std::vector<size_t>> out;
		split_recursive(group, graph, 0, out);
		return out;
	}

private:
	std::shared_ptr<GraphSplitCriterion> criterion_;
	int max_depth_;

	void split_recursive(std::vector<size_t> group, const ObjectGraph& graph, int depth,
	                      std::vector<std::vector<size_t>>& out) const {
		bool leaf = group.size() < 2 || depth >= max_depth_ || !criterion_->should_split(group, graph);
		if (leaf) {
			out.push_back(std::move(group));
			return;
		}

		GraphMinCut cut = graph_min_cut(group, graph);
		split_recursive(std::move(cut.side_a), graph, depth + 1, out);
		split_recursive(std::move(cut.side_b), graph, depth + 1, out);
	}
};
