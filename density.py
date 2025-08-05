import numpy as np
import pyvista as pv

pv.set_jupyter_backend("static")

# Define the grid dimensions
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
z = np.linspace(-5, 5, 50)
X, Y, Z = np.meshgrid(x, y, z)

# Define a 3D Gaussian density function
sigma = 1.0
density = np.exp(-(X**2 + 2 * Y**2 + Z**2) / (2 * sigma**2))


# |%%--%%| <3MT5TPuIUN|IvZkO7WzGZ>


# Create a PyVista grid
grid = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing="ij"))
grid["Density"] = density.ravel()
# Plot the density function
plotter = pv.Plotter()
plotter.add_volume(grid, scalars="Density", cmap="viridis", opacity="sigmoid")
plotter.show()
# |%%--%%| <IvZkO7WzGZ|2OuLrFatVC>

# Seed rng for reproducibility
rng = np.random.default_rng(seed=0)
points = rng.random((1000, 3))

pv.plot(
    points,
    scalars=points[:, 2],
    render_points_as_spheres=True,
    point_size=20,
    show_scalar_bar=False,
)
