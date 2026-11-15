#pragma once

// Object-level adaptation of DTSCAN (dtscan.hpp) for a collection of
// already-detected, individuated objects (e.g. berries from an ML model)
// instead of raw points. This file is NOT part of DTSCAN itself -- DTSCAN
// (as published, Kim & Cho 2019) is a plain point-clustering algorithm with
// no notion of "objects"; dtscan.hpp implements exactly that, faithfully.
// This is a thin layer on top, specific to this project's pipeline:
//
//   1. Each object contributes one or more points: a single centroid
//      (coarse/fast mode), or its own full contour (shape-aware mode) --
//      both use the exact same code path/signature here, since a centroid
//      is just a size-1 point set. Every object's points are pooled and run
//      through dtscan() exactly as published, with each point remembering
//      which object it came from.
//   2. Each object's cluster membership is the majority vote of its own
//      points' point-level DTSCAN labels (ties broken by the smallest label
//      id). An object whose points are entirely noise-labeled becomes its
//      own singleton group, rather than being silently dropped.
//
// Separately, this file also builds the *unfiltered* object-level proximity
// graph (every inter-object Delaunay edge and its own raw distance, with
// same-object edges ignored -- they say nothing about inter-object
// distance, and in full-contour mode would dominate and skew distances
// purely from within-object point density). This graph is independent of
// DTSCAN's own z-score pruning: it exists for GraphBottleneckSplit
// (graph_bottleneck_split.hpp) to further refine any DTSCAN cluster that
// still looks like multiple merged objects, exactly mirroring how
// BottleneckSplit's own find_bottleneck operates on a polygon's full
// boundary without any pre-filtering beyond the polygon it's already been
// handed.

#include <algorithm>
#include <cstddef>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "concavehull.hpp"  // next_halfedge, delaunator::Delaunator
#include "dtscan.hpp"
#include "split_criterion.hpp"  // Point2D
#include "thirdparty/delaunator-header-only.hpp"

// Weighted proximity graph over objects: adjacency[i] lists every object j
// with at least one inter-object Delaunay edge to i, paired with the
// shortest such edge's Euclidean length.
struct ObjectGraph {
	size_t n_objects = 0;
	std::vector<std::vector<std::pair<size_t, double>>> adjacency;
};

class ObjectClusterGraph {
public:
	// `object_points[k]` is object k's own point set (size 1 for a centroid
	// -only mode; its full contour for a shape-aware mode -- any non-empty
	// point set works identically). `area_z_threshold`/`length_z_threshold`/
	// `min_pts` are passed straight through to dtscan() over the pooled,
	// object-tagged points.
	ObjectClusterGraph(const std::vector<std::vector<Point2D>>& object_points, double area_z_threshold = 1.0,
	                    double length_z_threshold = 1.0, size_t min_pts = 3);

	const ObjectGraph& graph() const { return graph_; }

	// DTSCAN clustering, aggregated to object level by majority vote over
	// each object's own points' labels. Each inner vector is a list of
	// object indices (into the original `object_points`), covering every
	// object exactly once.
	const std::vector<std::vector<size_t>>& clusters() const { return clusters_; }

private:
	ObjectGraph graph_;
	std::vector<std::vector<size_t>> clusters_;
};

inline ObjectClusterGraph::ObjectClusterGraph(const std::vector<std::vector<Point2D>>& object_points,
                                               double area_z_threshold, double length_z_threshold,
                                               size_t min_pts) {
	size_t n_objects = object_points.size();
	graph_.n_objects = n_objects;
	graph_.adjacency.assign(n_objects, {});
	if (n_objects == 0) return;

	std::vector<Point2D> pooled;
	std::vector<size_t> point_object;
	for (size_t k = 0; k < n_objects; ++k) {
		for (const Point2D& p : object_points[k]) {
			pooled.push_back(p);
			point_object.push_back(k);
		}
	}

	// --- object-level DTSCAN clustering (majority vote over point labels) ---
	DTScanResult dt = dtscan(pooled, area_z_threshold, length_z_threshold, min_pts);
	std::vector<std::map<int, size_t>> votes(n_objects);  // object -> {label: count}
	for (size_t p = 0; p < pooled.size(); ++p) {
		if (dt.labels[p] != -1) ++votes[point_object[p]][dt.labels[p]];
	}
	std::map<int, std::vector<size_t>> groups;  // dtscan label -> object indices
	int next_singleton_id = -1;  // negative ids for objects with no clustered points
	for (size_t k = 0; k < n_objects; ++k) {
		if (votes[k].empty()) {
			groups[next_singleton_id--] = {k};
			continue;
		}
		int best_label = votes[k].begin()->first;
		size_t best_count = 0;
		for (auto& kv : votes[k]) {
			if (kv.second > best_count) {
				best_count = kv.second;
				best_label = kv.first;
			}
		}
		groups[best_label].push_back(k);
	}
	clusters_.reserve(groups.size());
	for (auto& kv : groups) clusters_.push_back(std::move(kv.second));

	// --- unfiltered object-level proximity graph (for GraphBottleneckSplit) ---
	if (pooled.size() < 3) return;  // delaunator needs >= 3 points total
	std::vector<double> coords;
	coords.reserve(pooled.size() * 2);
	for (auto& p : pooled) {
		coords.push_back(p.x);
		coords.push_back(p.y);
	}
	std::span<double> coords_span(coords);
	delaunator::Delaunator d(coords_span);
	size_t n_triangles = d.triangles.size() / 3;

	std::map<std::pair<size_t, size_t>, double> pair_distance;
	for (size_t t = 0; t < n_triangles; ++t) {
		for (int k = 0; k < 3; ++k) {
			size_t he = 3 * t + k;
			size_t twin = d.halfedges[he];
			if (!(twin == delaunator::INVALID_INDEX || he < twin)) continue;  // each undirected edge once

			size_t pa = d.triangles[he];
			size_t pb = d.triangles[next_halfedge(he)];
			size_t oa = point_object[pa], ob = point_object[pb];
			if (oa == ob) continue;

			double dx = coords[2 * pa] - coords[2 * pb];
			double dy = coords[2 * pa + 1] - coords[2 * pb + 1];
			double len = std::sqrt(dx * dx + dy * dy);

			auto key = oa < ob ? std::make_pair(oa, ob) : std::make_pair(ob, oa);
			auto it = pair_distance.find(key);
			if (it == pair_distance.end() || len < it->second) pair_distance[key] = len;
		}
	}

	for (auto& kv : pair_distance) {
		size_t a = kv.first.first, b = kv.first.second;
		double len = kv.second;
		graph_.adjacency[a].push_back({b, len});
		graph_.adjacency[b].push_back({a, len});
	}
}
