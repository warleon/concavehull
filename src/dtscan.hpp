#pragma once

// A faithful implementation of DTSCAN (Delaunay Triangulation-based Spatial
// Clustering of Applications with Noise):
//
//   Kim, J., & Cho, J. (2019). Delaunay Triangulation-Based Spatial
//   Clustering Technique for Enhanced Adjacent Boundary Detection and
//   Segmentation of LiDAR 3D Point Clouds. Sensors, 19(18), 3926.
//
// Transcribed directly from the paper's Section 3 (read from the PDF, not
// reconstructed from secondary summaries). Two stages:
//
//   1. z-score pruning (Eqs. 2-3). Build the Delaunay triangulation as an
//      undirected graph. Compute the mean/std of every triangle's AREA and
//      every edge's LENGTH, each over the WHOLE triangulation (the paper
//      computes these globally, not per-region: "the mean and variance of
//      ... the areas ... [and] the edge lengths ... of all of the triangles
//      are computed"). Each triangle gets a z-score for its own area and
//      for its longest edge's length; a triangle is discarded ("the
//      unnecessary components are sequentially removed by selecting an
//      appropriate threshold value") if either z-score exceeds its own
//      threshold. The paper, like chi-shape's own chi parameter, does not
//      prescribe one fixed universal threshold -- it is data/application
//      dependent, exposed here as area_z_threshold/length_z_threshold.
//
//   2. DBSCAN's own region-growing mechanism (Ester et al. 1996) over the
//      surviving edges, using triangulation-adjacency as the neighbor
//      relation in place of DBSCAN's usual fixed-radius ball query -- this
//      substitution is DTSCAN's whole contribution. Quoting the paper's own
//      "clustering process of DTSCAN", verbatim:
//        1. A neighboring node of point pi in a triangle is a set of edge
//           graphs directly connected to pi.
//        2. If the number of elements in the set of connected neighboring
//           nodes is greater than or equal to MinPts, it is included as one
//           cluster, Cq.
//        3. (1)-(2) is repeated for neighboring nodes of the point, and the
//           cluster is expanded by searching for neighboring nodes.
//        4. Points without connection nodes are classified as noise or
//           anomalous points.
//
// This file implements DTSCAN exactly as published, operating on raw
// points -- it has no notion of "objects" with multiple points each. The
// object-aware adaptation used elsewhere in this project (grouping
// already-detected objects, e.g. berries, each contributing one or more of
// its own points) is a separate, clearly-distinct layer built ON TOP of
// this -- see object_cluster.hpp -- not part of DTSCAN itself.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "concavehull.hpp"  // next_halfedge, edge_length, delaunator::Delaunator
#include "split_criterion.hpp"  // Point2D
#include "thirdparty/delaunator-header-only.hpp"

struct DTScanResult {
	// labels[i] is the 0-based cluster id of points[i], or -1 if noise.
	std::vector<int> labels;
	size_t n_clusters = 0;
};

inline double dtscan_mean(const std::vector<double>& v) {
	if (v.empty()) return 0.0;
	double s = 0.0;
	for (double x : v) s += x;
	return s / v.size();
}

inline double dtscan_stddev(const std::vector<double>& v, double mean) {
	if (v.empty()) return 0.0;
	double s = 0.0;
	for (double x : v) s += (x - mean) * (x - mean);
	return std::sqrt(s / v.size());
}

inline DTScanResult dtscan(const std::vector<Point2D>& points, double area_z_threshold = 1.0,
                            double length_z_threshold = 1.0, size_t min_pts = 3) {
	size_t n = points.size();
	DTScanResult result;
	result.labels.assign(n, -1);
	if (n < 3) return result;  // delaunator needs >= 3 points; fewer is trivially all noise

	std::vector<double> coords;
	coords.reserve(n * 2);
	for (auto& p : points) {
		coords.push_back(p.x);
		coords.push_back(p.y);
	}
	std::span<double> coords_span(coords);
	delaunator::Delaunator d(coords_span);
	size_t n_triangles = d.triangles.size() / 3;
	if (n_triangles == 0) return result;

	// --- Stage 1: z-score pruning (Eqs. 2-3) ---
	std::vector<double> areas(n_triangles);
	for (size_t t = 0; t < n_triangles; ++t) {
		size_t pa = d.triangles[3 * t], pb = d.triangles[3 * t + 1], pc = d.triangles[3 * t + 2];
		double ax = coords[2 * pa], ay = coords[2 * pa + 1];
		double bx = coords[2 * pb], by = coords[2 * pb + 1];
		double cx = coords[2 * pc], cy = coords[2 * pc + 1];
		areas[t] = std::abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2.0;
	}
	double area_mean = dtscan_mean(areas);
	double area_std = dtscan_stddev(areas, area_mean);

	std::vector<double> he_length(3 * n_triangles);
	std::vector<double> unique_lengths;
	for (size_t he = 0; he < 3 * n_triangles; ++he) {
		he_length[he] = edge_length(d, he);
		size_t twin = d.halfedges[he];
		if (twin == delaunator::INVALID_INDEX || he < twin) unique_lengths.push_back(he_length[he]);
	}
	double length_mean = dtscan_mean(unique_lengths);
	double length_std = dtscan_stddev(unique_lengths, length_mean);

	std::vector<bool> removed_triangle(n_triangles, false);
	for (size_t t = 0; t < n_triangles; ++t) {
		double area_z = area_std > 0.0 ? (areas[t] - area_mean) / area_std : 0.0;
		double max_len_z = -std::numeric_limits<double>::infinity();
		for (int k = 0; k < 3; ++k) {
			double len = he_length[3 * t + k];
			double z = length_std > 0.0 ? (len - length_mean) / length_std : 0.0;
			max_len_z = std::max(max_len_z, z);
		}
		if (area_z > area_z_threshold || max_len_z > length_z_threshold) removed_triangle[t] = true;
	}

	// Pruned adjacency: edges of every surviving triangle only.
	std::vector<std::vector<size_t>> neighbors(n);
	for (size_t t = 0; t < n_triangles; ++t) {
		if (removed_triangle[t]) continue;
		for (int k = 0; k < 3; ++k) {
			size_t he = 3 * t + k;
			size_t pa = d.triangles[he];
			size_t pb = d.triangles[next_halfedge(he)];
			neighbors[pa].push_back(pb);
			neighbors[pb].push_back(pa);
		}
	}
	for (auto& nb : neighbors) {
		std::sort(nb.begin(), nb.end());
		nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
	}

	// --- Stage 2: DBSCAN region growing (Ester et al. 1996) over
	// triangulation-adjacency, exactly as DTSCAN's own 4-step description
	// above (neighbor = triangulation-adjacent; MinPts; expand via core
	// points; no-connection points are noise). ---
	std::vector<bool> visited(n, false);
	int cluster_id = 0;
	for (size_t i = 0; i < n; ++i) {
		if (visited[i]) continue;
		visited[i] = true;
		if (neighbors[i].size() < min_pts) continue;  // not a core point -- tentatively noise

		result.labels[i] = cluster_id;
		std::vector<size_t> seeds(neighbors[i].begin(), neighbors[i].end());
		for (size_t idx = 0; idx < seeds.size(); ++idx) {
			size_t q = seeds[idx];
			if (!visited[q]) {
				visited[q] = true;
				if (neighbors[q].size() >= min_pts) {
					seeds.insert(seeds.end(), neighbors[q].begin(), neighbors[q].end());
				}
			}
			if (result.labels[q] == -1) result.labels[q] = cluster_id;
		}
		++cluster_id;
	}
	result.n_clusters = (size_t)cluster_id;
	return result;
}
