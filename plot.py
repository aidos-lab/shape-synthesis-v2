import pyvista as pv

pv.global_theme.volume_mapper = "smart"

model = pv.Wavelet()

p = pv.Plotter(notebook=False)
p.add_volume(model, mapper="smart")
p.add_mesh(model.outline(), color="white")
p.show()
