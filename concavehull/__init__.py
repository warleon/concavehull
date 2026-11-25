"""concavehull: concave hull / chi-outline / bottleneck-split algorithms
for 2D point sets (pybind11 extension, see `_concavehull`), plus a pure
-Python `three_d` submodule for the 3D generalizations that need a 3D
Delaunay triangulation (via GUDHI) rather than the 2D-only triangulator
the compiled extension uses internally.
"""

from ._concavehull import *  # noqa: F401,F403

from . import three_d  # noqa: F401
