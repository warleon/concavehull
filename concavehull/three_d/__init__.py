"""3D generalizations of this project's algorithms that need a true 3D
Delaunay triangulation (via GUDHI) rather than the 2D-only triangulator
the compiled `concavehull._concavehull` extension uses.
"""

from .dtscan_3d import DTScan3DResult, dtscan_3d  # noqa: F401
