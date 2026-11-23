#include <cstring>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

#include "alpha_outline.hpp"
#include "bottleneck.hpp"
#include "chioutline.hpp"
#include "concavehull.hpp"
#include "dtscan.hpp"
#include "graph.hpp"
#include "graph_bottleneck_split.hpp"
#include "graph_callables.hpp"
#include "object_cluster.hpp"
#include "split_criterion.hpp"

namespace py = pybind11;

std::vector<py::array_t<double>> polygons_to_py(const std::vector<std::vector<double>>& polygons) {
	std::vector<py::array_t<double>> result;
	result.reserve(polygons.size());

	for (auto& poly : polygons) {
		py::array_t<double> arr({(py::ssize_t)(poly.size() / 2), (py::ssize_t)2});
		std::memcpy(arr.mutable_data(), poly.data(), poly.size() * sizeof(double));
		result.push_back(std::move(arr));
	}

	return result;
}

std::vector<double> py_polygon_to_flat(const py::array_t<double>& poly) {
	py::buffer_info buf = poly.request();
	if (buf.ndim != 2) {
		throw std::runtime_error("Number of dimensions must be two");
	}
	if (buf.shape[1] != 2) {
		throw std::runtime_error("Second dimension's size must be two");
	}
	double *ptr = static_cast<double *>(buf.ptr);
	return std::vector<double>(ptr, ptr + buf.size);
}

std::vector<Point2D> py_polygon_to_points(const py::array_t<double>& poly) {
	std::vector<double> flat = py_polygon_to_flat(poly);
	std::vector<Point2D> pts;
	pts.reserve(flat.size() / 2);
	for (size_t i = 0; i + 1 < flat.size(); i += 2) {
		pts.push_back({flat[i], flat[i + 1]});
	}
	return pts;
}

py::array_t<double> pyconcavehull(const py::array_t<double>& pts, double chi_factor) {
	py::buffer_info buf = pts.request();
	
	if (buf.ndim != 2) {
		throw std::runtime_error("Number of dimensions must be two");
	}
	
	if (buf.shape[1] != 2) {
		throw std::runtime_error("Second dimension's size must be two");
	}

	double *ptr = static_cast<double *>(buf.ptr);

	std::vector<double> ch = concavehull({ptr, ptr+buf.size}, chi_factor);

	py::buffer_info resbuf = py::buffer_info(
		ch.data(),
		buf.itemsize,
		buf.format,
		buf.ndim,
		{(size_t)(ch.size() >> 1), (size_t)2},
		buf.strides
		);
	return py::array_t<double>(resbuf);
}

Graph pyconcavehull_graph(const py::array_t<double>& pts, double chi_factor) {
	py::buffer_info buf = pts.request();
	if (buf.ndim != 2) {
		throw std::runtime_error("Number of dimensions must be two");
	}
	if (buf.shape[1] != 2) {
		throw std::runtime_error("Second dimension's size must be two");
	}
	double *ptr = static_cast<double *>(buf.ptr);
	return concavehull_graph({ptr, ptr + buf.size}, chi_factor);
}

std::unique_ptr<ChiOutline> make_chioutline(const py::array_t<double>& pts, double chi) {
	py::buffer_info buf = pts.request();

	if (buf.ndim != 2) {
		throw std::runtime_error("Number of dimensions must be two");
	}

	if (buf.shape[1] != 2) {
		throw std::runtime_error("Second dimension's size must be two");
	}

	double *ptr = static_cast<double *>(buf.ptr);
	std::vector<double> coords(ptr, ptr + buf.size);

	// Constructed directly in its final heap location and handed to pybind11
	// as a unique_ptr: ChiOutline is non-movable (see chioutline.hpp), so it
	// must never be returned by value from this factory.
	return std::make_unique<ChiOutline>(coords, chi);
}

std::vector<py::array_t<double>> pychioutline_compute(ChiOutline& self) {
	return polygons_to_py(self.compute());
}

double pychioutline_length_for_chi(const py::array_t<double>& pts, double chi) {
	return chi_outline_length_for_chi(py_polygon_to_flat(pts), chi);
}

double pychioutline_chi_for_length(const py::array_t<double>& pts, double length) {
	return chi_outline_chi_for_length(py_polygon_to_flat(pts), length);
}

double pychioutline_max_meaningful_chi(const py::array_t<double>& pts) {
	return chi_outline_max_meaningful_chi(py_polygon_to_flat(pts));
}

double pychioutline_elbow_chi(const py::array_t<double>& pts) {
	return chi_outline_elbow_chi(py_polygon_to_flat(pts));
}

py::dict polygon_with_holes_to_py(const PolygonWithHoles& pwh) {
	py::dict d;
	d["exterior"] = polygons_to_py({pwh.exterior})[0];
	py::list holes_list;
	for (auto& h : polygons_to_py(pwh.holes)) holes_list.append(h);
	d["holes"] = holes_list;
	return d;
}

std::unique_ptr<AlphaOutline> make_alpha_outline(const py::array_t<double>& pts, double chi) {
	py::buffer_info buf = pts.request();

	if (buf.ndim != 2) {
		throw std::runtime_error("Number of dimensions must be two");
	}

	if (buf.shape[1] != 2) {
		throw std::runtime_error("Second dimension's size must be two");
	}

	double *ptr = static_cast<double *>(buf.ptr);
	std::vector<double> coords(ptr, ptr + buf.size);

	// Same non-movable-must-construct-in-place reasoning as ChiOutline.
	return std::make_unique<AlphaOutline>(coords, chi);
}

std::vector<py::dict> pyalpha_outline_compute(const AlphaOutline& self) {
	std::vector<py::dict> out;
	for (auto& pwh : self.compute()) out.push_back(polygon_with_holes_to_py(pwh));
	return out;
}

std::vector<Graph> pychioutline_compute_graph(const ChiOutline& self) { return self.compute_graph(); }

std::vector<Graph> pyalpha_outline_compute_graph(const AlphaOutline& self) { return self.compute_graph(); }

Graph make_graph(const py::array_t<double>& vertices, const py::array_t<long long>& edges) {
	Graph g;
	g.vertices = py_polygon_to_points(vertices);
	py::buffer_info buf = edges.request();
	if (buf.ndim != 2 || buf.shape[1] != 2) {
		throw std::runtime_error("edges must be an Nx2 array");
	}
	long long *ptr = static_cast<long long *>(buf.ptr);
	for (py::ssize_t i = 0; i < buf.shape[0]; ++i) {
		g.edges.push_back({(size_t)ptr[2 * i], (size_t)ptr[2 * i + 1]});
	}
	return g;
}

py::array_t<double> graph_vertices_to_py(const Graph& g) {
	py::array_t<double> arr({(py::ssize_t)g.vertices.size(), (py::ssize_t)2});
	double *ptr = arr.mutable_data();
	for (size_t i = 0; i < g.vertices.size(); ++i) {
		ptr[2 * i] = g.vertices[i].x;
		ptr[2 * i + 1] = g.vertices[i].y;
	}
	return arr;
}

py::array_t<size_t> graph_edges_to_py(const Graph& g) {
	py::array_t<size_t> arr({(py::ssize_t)g.edges.size(), (py::ssize_t)2});
	size_t *ptr = arr.mutable_data();
	for (size_t i = 0; i < g.edges.size(); ++i) {
		ptr[2 * i] = g.edges[i].first;
		ptr[2 * i + 1] = g.edges[i].second;
	}
	return arr;
}

py::object graph_data_to_py(const std::vector<double>& data) {
	if (data.empty()) return py::none();
	py::array_t<double> arr((py::ssize_t)data.size());
	std::memcpy(arr.mutable_data(), data.data(), data.size() * sizeof(double));
	return arr;
}

VertexDataFn py_to_vertex_fn(const py::object& obj) {
	if (obj.is_none()) return nullptr;
	return obj.cast<std::function<std::vector<double>(const Graph&)>>();
}

EdgeDataFn py_to_edge_fn(const py::object& obj) {
	if (obj.is_none()) return nullptr;
	return obj.cast<std::function<std::vector<double>(const Graph&)>>();
}

EdgeWeightFn py_to_weight_fn(const py::object& obj) {
	if (obj.is_none()) return euclidean_edge_weight;
	return obj.cast<std::function<double(const Graph&, size_t)>>();
}

void pyannotate_graph(Graph& g, const py::object& vertex_fn, const py::object& edge_fn) {
	annotate_graph(g, py_to_vertex_fn(vertex_fn), py_to_edge_fn(edge_fn));
}

py::dict pygraph_piece_to_py(const GraphPiece& piece) {
	py::dict d;
	d["graph"] = piece.graph;
	d["boundary"] = piece.boundary;
	return d;
}

std::vector<py::dict> pybottleneck_split_graph(const BottleneckSplit& self, const Graph& g,
                                                 const std::vector<size_t>& boundary, const py::object& weight_fn) {
	auto pieces = self.split_graph(g, boundary, py_to_weight_fn(weight_fn));
	std::vector<py::dict> out;
	for (auto& piece : pieces) out.push_back(pygraph_piece_to_py(piece));
	return out;
}

std::vector<py::array_t<double>> pybottleneck_split(const BottleneckSplit& self, const py::array_t<double>& polygon) {
	return polygons_to_py(self.split(py_polygon_to_flat(polygon)));
}

std::vector<py::array_t<double>> pybottleneck_split_all(const BottleneckSplit& self,
                                                          const std::vector<py::array_t<double>>& polygons) {
	std::vector<std::vector<double>> flats;
	flats.reserve(polygons.size());
	for (auto& poly : polygons) {
		flats.push_back(py_polygon_to_flat(poly));
	}
	return polygons_to_py(self.split_all(flats));
}

py::array_t<int> pydtscan(const py::array_t<double>& pts, double area_z_threshold, double length_z_threshold,
                           size_t min_pts) {
	std::vector<Point2D> points = py_polygon_to_points(pts);
	DTScanResult r = dtscan(points, area_z_threshold, length_z_threshold, min_pts);
	py::array_t<int> labels((py::ssize_t)r.labels.size());
	std::memcpy(labels.mutable_data(), r.labels.data(), r.labels.size() * sizeof(int));
	return labels;
}

std::vector<std::vector<Point2D>> py_object_points_to_points(const std::vector<py::array_t<double>>& object_points) {
	std::vector<std::vector<Point2D>> out;
	out.reserve(object_points.size());
	for (auto& pts : object_points) out.push_back(py_polygon_to_points(pts));
	return out;
}

std::unique_ptr<ObjectClusterGraph> make_object_cluster_graph(const std::vector<py::array_t<double>>& object_points,
                                                                double area_z_threshold, double length_z_threshold,
                                                                size_t min_pts) {
	return std::make_unique<ObjectClusterGraph>(py_object_points_to_points(object_points), area_z_threshold,
	                                             length_z_threshold, min_pts);
}

std::vector<std::vector<size_t>> pyobject_cluster_graph_clusters(const ObjectClusterGraph& self) {
	return self.clusters();
}

std::vector<std::vector<size_t>> pygraph_bottleneck_split(const GraphBottleneckSplit& self,
                                                            const std::vector<size_t>& group,
                                                            const ObjectClusterGraph& cluster_graph) {
	return self.split(group, cluster_graph.graph());
}

PYBIND11_MODULE(concavehull, m) {
	m.def("concavehull", &pyconcavehull, "Find concave hull from array of 2D points",
	      py::arg("pts"), py::arg("chi_factor") = 0.1);

	m.def("concavehull_graph", &pyconcavehull_graph, py::arg("pts"), py::arg("chi_factor") = 0.1,
	      "\"The complete graph\" for concavehull(): the Delaunay triangulation "
	      "of every input point, MINUS every triangle the dig excluded (see "
	      "concavehull_impl's own comment in concavehull.hpp) -- deliberately "
	      "depends on chi_factor (more digging excludes more triangles), "
	      "unlike returning the raw unfiltered triangulation would.");

	py::class_<Graph>(m, "Graph",
	    "A plain undirected graph over polygon/point-cloud vertices -- the "
	    "full surviving mesh of whichever outline algorithm produced it "
	    "(concavehull_graph(), ChiOutline.compute_graph(), "
	    "AlphaOutline.compute_graph()), not just its boundary cycle. "
	    "Optionally annotated with custom per-vertex/per-edge data via "
	    "annotate_graph(), which BottleneckSplit.split_graph() can then use "
	    "(via a weight_fn callable) in place of Euclidean edge length.")
	    .def(py::init(&make_graph), py::arg("vertices"), py::arg("edges"),
	         "Hand-build a Graph directly: `vertices` is an Nx2 array, "
	         "`edges` is an Mx2 integer array of vertex-index pairs.")
	    .def_property_readonly("vertices", &graph_vertices_to_py, "Nx2 array of vertex coordinates.")
	    .def_property_readonly("edges", &graph_edges_to_py, "Mx2 integer array of vertex-index pairs.")
	    .def_property_readonly(
	        "vertex_data", [](const Graph& g) { return graph_data_to_py(g.vertex_data); },
	        "Per-vertex custom data (array, same length as vertices), or None if never annotated.")
	    .def_property_readonly(
	        "edge_data", [](const Graph& g) { return graph_data_to_py(g.edge_data); },
	        "Per-edge custom data (array, same length as edges), or None if never annotated.");

	m.def("annotate_graph", &pyannotate_graph, py::arg("graph"), py::arg("vertex_fn") = py::none(),
	      py::arg("edge_fn") = py::none(),
	      "Mutates `graph` in place: `vertex_fn(graph) -> list[float]` (one "
	      "value per vertex) runs first and is stored as graph.vertex_data; "
	      "`edge_fn(graph) -> list[float]` (one value per edge, may read the "
	      "just-populated graph.vertex_data) runs second and is stored as "
	      "graph.edge_data. Either may be omitted (None) to skip that half. "
	      "Any Python callable works here -- a plain function, a lambda, or "
	      "an object with __call__.");

	m.def("vertex_degree", [](const Graph& g) { return vertex_degree(g); }, py::arg("graph"),
	      "VertexDataFn: each vertex's own incident-edge count.");
	m.def(
	    "edge_weight_sum_of_vertex_data", [](const Graph& g) { return edge_weight_sum_of_vertex_data(g); },
	    py::arg("graph"),
	    "EdgeDataFn: each edge's weight = the sum of its two endpoints' "
	    "already-populated graph.vertex_data. Composed with vertex_degree via "
	    "annotate_graph(g, vertex_degree, edge_weight_sum_of_vertex_data), "
	    "the resulting graph.edge_data is then usable directly as a weight_fn "
	    "for BottleneckSplit.split_graph() -- `lambda g, e: g.edge_data[e]`.");

	py::class_<ChiOutline>(m, "ChiOutline",
	    "Chi-outline algorithm (Zhong & Duckham, 2016): extends the chi-shape "
	    "algorithm to produce possibly disconnected simple polygons and to "
	    "exclude outliers, using a single parameter chi. NOTE: chi here is NOT "
	    "the same kind of quantity as concavehull()'s chi_factor -- chi_factor "
	    "is normalized to [0, 1], but chi is an unbounded multiplier on the "
	    "edge-length standard deviation (lambda = mu + chi*sigma, Eq. 4 of the "
	    "paper); the two papers just reuse the same Greek letter for "
	    "differently-scaled parameters. Use length_to_chi()/chi_to_length() to "
	    "convert to/from an actual edge-length threshold for a given point set, "
	    "and max_meaningful_chi() to find where chi saturates to the convex hull.")
	    .def(py::init(&make_chioutline), py::arg("pts"), py::arg("chi") = 0.1)
	    .def("compute", &pychioutline_compute,
	         "Run the algorithm, returning a list of Nx2 arrays, one per "
	         "disconnected polygon found.")
	    .def("compute_graph", &pychioutline_compute_graph,
	         "'The complete graph' version of compute(): one Graph per "
	         "disconnected region (same components/filter as compute()), "
	         "containing every vertex and edge of that region's own kept "
	         "triangulation, not just its boundary cycle. Must be called "
	         "exactly once, directly after compute() -- it reads the final "
	         "state compute() leaves behind, the same one-shot restriction "
	         "compute() itself already has.")
	    .def_static("chi_to_length", &pychioutline_length_for_chi, py::arg("pts"), py::arg("chi"),
	         "Convert chi to the corresponding edge-length threshold "
	         "lambda = mu + chi*sigma for `pts`'s whole Delaunay triangulation "
	         "(Eq. 4, at chi-outline's initial single-component state).")
	    .def_static("length_to_chi", &pychioutline_chi_for_length, py::arg("pts"), py::arg("length"),
	         "Inverse of chi_to_length(): the chi value whose edge-length "
	         "threshold for `pts` equals `length`.")
	    .def_static("max_meaningful_chi", &pychioutline_max_meaningful_chi, py::arg("pts"),
	         "The data-dependent chi value at/beyond which chi-outline "
	         "saturates to the convex hull of `pts`, per the paper's own bound "
	         "chi >= (max(DT_b) - mu_DT) / sigma_DT.")
	    .def_static("elbow_chi", &pychioutline_elbow_chi, py::arg("pts"),
	         "A default chi derived from pts's own Delaunay edge-length "
	         "distribution: finds the 'elbow' separating short (within-cluster) "
	         "edges from long (between-cluster/outlier) ones via the maximum-"
	         "distance-from-chord method on the sorted length curve, then "
	         "converts that length to chi. Unlike a heuristic based on some "
	         "externally-supplied scale (e.g. an object's bounding-box size), "
	         "this can't be mismatched against this point cloud's own spacing.");

	py::class_<AlphaOutline>(m, "AlphaOutline",
	    "Delaunay-triangulation-based outline extraction (alpha-shape family, "
	    "Edelsbrunner, Kirkpatrick & Seidel 1983; one-shot triangulate-filter-"
	    "extract pipeline in the same shape as Polylidar, Castagno & Atkins "
	    "2020) that keeps interior holes instead of forcing a single simple "
	    "polygon per region the way ChiOutline deliberately does. Same "
	    "lambda = mu + chi*sigma threshold formula/parameter as ChiOutline, "
	    "for familiarity, but estimated once globally -- unlike ChiOutline, "
	    "this does NOT adapt to very different point densities within a "
	    "single call. A single global chi can also carve small spurious "
	    "holes out of a dense-but-not-perfectly-uniform blob (a known "
	    "characteristic of alpha shapes in general, not a bug) -- tune chi "
	    "down, or prefer ChiOutline, when that matters more than hole support.")
	    .def(py::init(&make_alpha_outline), py::arg("pts"), py::arg("chi") = 0.1)
	    .def("compute", &pyalpha_outline_compute,
	         "Run the algorithm, returning a list of "
	         "{'exterior': Nx2 array, 'holes': list of Nx2 arrays} dicts, "
	         "one per disconnected region.")
	    .def("compute_graph", &pyalpha_outline_compute_graph,
	         "'The complete graph' version of compute(): one Graph per "
	         "disconnected region, containing every vertex and edge of that "
	         "region's own kept triangulation -- not just its boundary/hole "
	         "cycles. Repeatable (like compute()).");

	py::class_<SplitCriterion, std::shared_ptr<SplitCriterion>>(m, "SplitCriterion",
	    "Abstract shape-classification criterion: decides whether a polygon "
	    "still looks like a clump that should be split. Pass a concrete "
	    "subclass instance to BottleneckSplit, or call should_split() "
	    "directly to inspect/tune a criterion on its own.")
	    .def("should_split", [](const SplitCriterion& self, const py::array_t<double>& polygon) {
	        return self.should_split(py_polygon_to_points(polygon));
	    }, py::arg("polygon"));

	py::class_<ConvexHullL2Criterion, SplitCriterion, std::shared_ptr<ConvexHullL2Criterion>>(
	    m, "ConvexHullL2Criterion",
	    "Chi-shape paper's L2 error norm (Duckham et al. 2008, Sec. 6.1), applied "
	    "between a polygon and its own convex hull: "
	    "L2 = (area(hull) - area(polygon)) / area(polygon) = 1/solidity - 1. "
	    "NOT bounded to [0, 1] -- it is 0 for an already-convex polygon and "
	    "grows without bound as the polygon thins out relative to its hull. "
	    "See ConvexDeficitCriterion for a [0, 1)-bounded alternative. Solidity "
	    "is sensitive to the chi/chi_factor used to build the polygon, not "
	    "just to real merging -- use a moderate-to-high chi (~0.3-0.6) and "
	    "calibrate the threshold on representative examples.")
	    .def(py::init<double>(), py::arg("l2_threshold") = 0.18);

	py::class_<ConvexDeficitCriterion, SplitCriterion, std::shared_ptr<ConvexDeficitCriterion>>(
	    m, "ConvexDeficitCriterion",
	    "Recommended [0, 1)-bounded alternative to ConvexHullL2Criterion: "
	    "deficit = (area(hull) - area(polygon)) / area(hull) = 1 - solidity, "
	    "which is always in [0, 1) since a polygon's area can never exceed "
	    "its own convex hull's area. This is the default used by "
	    "BottleneckSplit when no criterion is supplied. Same calibration "
	    "caveat as ConvexHullL2Criterion applies: tune against your own data.")
	    .def(py::init<double>(), py::arg("deficit_threshold") = 0.15);

	py::class_<BottleneckCostCriterion, SplitCriterion, std::shared_ptr<BottleneckCostCriterion>>(
	    m, "BottleneckCostCriterion",
	    "Recommended default criterion: splits only if the best candidate "
	    "bottleneck pair's own cost Es(A,B) = dist(A,B)/min(arc(A,B),arc(B,A)) "
	    "drops below `cost_threshold`. Unlike the convex-hull-area criteria "
	    "above, this judges a single candidate neck rather than the whole "
	    "polygon's shape, so it is far less confused by ordinary boundary "
	    "noise elsewhere on the boundary. An adjacent vertex pair always "
	    "costs exactly 1.0, so thresholds should stay well under that. "
	    "Tested empirically to separate genuinely touching objects from "
	    "noisy single blobs only when the input polygon's chi/chi_factor is "
	    "~0.5 or higher -- at lower chi this (like the other criteria) is "
	    "unreliable regardless of threshold.")
	    .def(py::init<double, double>(), py::arg("cost_threshold") = 0.45, py::arg("min_arc_fraction") = 0.1);

	py::class_<BottleneckSplit>(m, "BottleneckSplit",
	    "Clump splitting via bottleneck detection (Wang, Zhang & Ray, 2012). "
	    "Recursively cuts a polygon at its narrowest 'neck' -- the pair of "
	    "boundary points minimizing dist(A,B) / min(arc(A,B), arc(B,A)) -- "
	    "for as long as `criterion` says the result still looks like a clump. "
	    "Feed it polygons built with a moderate-to-high chi/chi_factor "
	    "(~0.5+) -- see BottleneckCostCriterion's docstring for why. "
	    "`min_piece_area` (default 0.0, disabled) rejects any candidate cut "
	    "that would produce a piece smaller than this (same units as the "
	    "input polygon's own coordinates, e.g. pixels^2 for a mask-derived "
	    "polygon) -- useful because the geometric criteria can mistake a "
	    "single object's own organic boundary lobes (e.g. individual grape "
	    "berries bulging within one mask contour) for real necks, otherwise "
	    "chipping it into many near-zero-area sliver pieces.")
	    .def(py::init<std::shared_ptr<SplitCriterion>, double, int, double>(),
	         py::arg("criterion") = nullptr, py::arg("min_arc_fraction") = 0.1, py::arg("max_depth") = 12,
	         py::arg("min_piece_area") = 0.0)
	    .def("split", &pybottleneck_split, py::arg("polygon"),
	         "Split a single polygon (Nx2 array) into sub-polygons.")
	    .def("split_all", &pybottleneck_split_all, py::arg("polygons"),
	         "Split every polygon in a list (e.g. ChiOutline.compute() output) "
	         "and return the flattened result.")
	    .def("split_graph", &pybottleneck_split_graph, py::arg("graph"), py::arg("boundary"),
	         py::arg("weight_fn") = py::none(),
	         "Graph-accepting generalization of split(): `graph` is a complete "
	         "mesh (e.g. from ChiOutline.compute_graph()/AlphaOutline."
	         "compute_graph()/concavehull_graph()), `boundary` is its own "
	         "boundary cycle (list of vertex indices into graph.vertices, "
	         "cyclic order -- pair the compute_graph() output with the "
	         "corresponding compute()/concavehull() output's own vertex "
	         "order, matched by coordinates). `weight_fn(graph, edge_idx) -> "
	         "float` (default None = Euclidean edge length, so omitting it "
	         "reproduces split()'s own numbers when `graph` is that "
	         "polygon's triangulation) replaces distance everywhere -- both "
	         "for the bottleneck cost and for the A* cut path, which is "
	         "constrained strictly to graph's own edges rather than any "
	         "injected straight-line chord. Returns a list of "
	         "{'graph': Graph, 'boundary': list[int]} dicts, one per "
	         "resulting piece.");

	m.def("dtscan", &pydtscan, py::arg("pts"), py::arg("area_z_threshold") = 1.0,
	      py::arg("length_z_threshold") = 1.0, py::arg("min_pts") = 3,
	      "DTSCAN (Kim & Cho, 2019, Sensors 19(18), 3926), implemented exactly "
	      "as published: builds the Delaunay triangulation, discards triangles "
	      "whose own area z-score or longest-edge length z-score (each "
	      "computed globally, over the WHOLE triangulation) exceeds "
	      "area_z_threshold/length_z_threshold, then runs DBSCAN's own "
	      "region-growing over the survivors using triangulation-adjacency as "
	      "the neighbor relation in place of DBSCAN's usual fixed-radius ball "
	      "query (min_pts is DBSCAN's MinPts). Returns one cluster label per "
	      "input point (-1 for noise/unclustered points). A plain point "
	      "-clustering algorithm with no notion of objects -- see "
	      "ObjectClusterGraph for the object-aware adaptation used elsewhere "
	      "in this package.");

	py::class_<ObjectClusterGraph>(m, "ObjectClusterGraph",
	    "Object-level adaptation of dtscan() for a collection of already"
	    "-detected, individuated objects (e.g. berries from an ML model) "
	    "instead of raw points -- NOT part of DTSCAN itself, which has no "
	    "notion of objects. `object_points[k]` is object k's own point set: "
	    "a single centroid (coarse/fast mode) or its own full contour "
	    "(shape-aware mode) both work identically, since a centroid is just "
	    "a size-1 point set. Every object's points are pooled, tagged by "
	    "source object, and run through dtscan() unmodified; each object's "
	    "cluster membership is the majority vote of its own points' labels "
	    "(an object with no clustered points becomes its own singleton "
	    "group, never silently dropped). Also builds the UNFILTERED "
	    "object-level proximity graph (every inter-object Delaunay edge's "
	    "own raw distance) that GraphBottleneckSplit needs to further refine "
	    "any cluster here that still looks like multiple merged objects.")
	    .def(py::init(&make_object_cluster_graph), py::arg("object_points"), py::arg("area_z_threshold") = 1.0,
	         py::arg("length_z_threshold") = 1.0, py::arg("min_pts") = 3)
	    .def("clusters", &pyobject_cluster_graph_clusters,
	         "The DTSCAN clustering, aggregated to object level: a list of "
	         "lists of object indices, covering every object exactly once.");

	py::class_<GraphSplitCriterion, std::shared_ptr<GraphSplitCriterion>>(m, "GraphSplitCriterion",
	    "Abstract criterion deciding whether a group of objects still looks "
	    "like multiple merged objects that GraphBottleneckSplit should cut "
	    "further. The graph-level analogue of SplitCriterion.")
	    .def("should_split", &GraphSplitCriterion::should_split, py::arg("group"), py::arg("graph"));

	py::class_<GraphBottleneckCostCriterion, GraphSplitCriterion, std::shared_ptr<GraphBottleneckCostCriterion>>(
	    m, "GraphBottleneckCostCriterion",
	    "Default criterion: cost = min_cut_weight / min(assoc(A,V), assoc(B,V)) "
	    "for the best normalized cut of the group's induced affinity graph "
	    "(affinity = 1/distance; assoc(X,V) = total affinity-weighted degree "
	    "of side X) -- the direct graph analogue of BottleneckCostCriterion's "
	    "arc-fraction-normalized cost, and scale-invariant the same way that "
	    "one is (see best_normalized_cut in the C++ source for why a plain "
	    "object-count normalization is NOT scale-invariant here). Splits only "
	    "if cost < cost_threshold AND both resulting sides have at least "
	    "min_group_fraction * |group| objects (the graph analogue of "
	    "min_arc_fraction).")
	    .def(py::init<double, double>(), py::arg("cost_threshold") = 0.1, py::arg("min_group_fraction") = 0.1);

	py::class_<GraphBottleneckSplit>(m, "GraphBottleneckSplit",
	    "Graph-level generalization of BottleneckSplit for a cluster of "
	    "already-detected, individuated objects (ObjectClusterGraph), "
	    "instead of a polygon boundary. Recursively cuts a group of object "
	    "indices at its narrowest graph connection -- a minimum cut of the "
	    "group's own induced affinity graph, found via best_normalized_cut "
	    "(stoer_wagner.hpp) rather than a plain global min cut (biased toward "
	    "isolating a tiny corner instead of a balanced separating cut) or a "
	    "cardinality-normalized ratio cut (not scale-invariant for an "
	    "affinity graph) -- for as long as `criterion` says the group still "
	    "looks merged. No polygon geometry is involved: output is purely a "
	    "partition of object indices, covering every object exactly once.")
	    .def(py::init<std::shared_ptr<GraphSplitCriterion>, int>(), py::arg("criterion") = nullptr,
	         py::arg("max_depth") = 12)
	    .def("split", &pygraph_bottleneck_split, py::arg("group"), py::arg("cluster_graph"),
	         "Recursively split `group` (a list of object indices into "
	         "`cluster_graph`) into final sub-groups, returned as a list of "
	         "lists of object indices.");
}
