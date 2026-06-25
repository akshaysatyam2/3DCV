import cv2
import numpy as np
import open3d as o3d

w_img, h_img = 800, 600
img = np.zeros((h_img, w_img, 3), dtype=np.uint8)

render = o3d.visualization.rendering.OffscreenRenderer(w_img, h_img)
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "unlitLine"
mat.line_width = 5.0

# coordinates: square at x=100..200, y=100..200, depth=100
points = [[100, 100, 100], [200, 100, 100], [200, 200, 100], [100, 200, 100]]
lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
colors = [[1, 0, 0] for _ in range(4)]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)

render.scene.add_geometry("box", line_set, mat)
render.scene.set_background([0, 0, 0, 1])

# to align exactly with pixel coords, focal length can be anything as long as depth is consistent
# let's just use f=1000
f = 1000
intrinsic = o3d.camera.PinholeCameraIntrinsic(w_img, h_img, f, f, w_img/2, h_img/2)
extrinsic = np.eye(4) # identity
render.setup_camera(intrinsic.intrinsic_matrix, extrinsic, w_img, h_img)
img_o3d = np.asarray(render.render_to_image())

print("max pixel value:", img_o3d.max())
cv2.imwrite("test_o3d.png", img_o3d)
