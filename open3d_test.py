import open3d as o3d
import numpy as np

vis3d = o3d.visualization.Visualizer()
vis3d.create_window() 
print(type(vis3d.get_render_option()))
opt = vis3d.get_render_option()
opt.background_color = np.asarray([0, 0, 0])
mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
mesh_sphere.compute_vertex_normals()
vis3d.add_geometry(mesh_sphere)
while True:
    vis3d.poll_events()
    vis3d.update_renderer()