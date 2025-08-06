import numpy as np
import pyvista as pv


# Seed rng for reproducibility
rng = np.random.default_rng(seed=0)
points = rng.random((10, 3))

pv.plot(points)
