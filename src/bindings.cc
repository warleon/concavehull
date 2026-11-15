#include <cstring>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

#include "alpha_outline.hpp"
#include "bottleneck.hpp"
#include "chioutline.hpp"
#include "concavehull.hpp"
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

PYBIND11_MODULE(concavehull, m) {
	m.def("concavehull", &pyconcavehull, "Find concave hull from array of 2D points",
	      py::arg("pts"), py::arg("chi_factor") = 0.1);

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
	         "one per disconnected region.");

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
	         "and return the flattened result.");
}
