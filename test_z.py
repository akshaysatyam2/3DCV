import open3d as o3d
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)
vc = vis.get_view_control()
print(hasattr(vc, 'set_constant_z_near'))
