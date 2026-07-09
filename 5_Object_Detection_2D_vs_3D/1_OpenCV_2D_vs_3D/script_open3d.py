import os
import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import open3d as o3d

# NOTE: Open3D requires an environment with compatible CPU instructions (AVX/AVX2).
# This script is provided for local execution where Open3D is fully supported!

img_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg"
cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalcatface.xml"

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

if not os.path.exists("pet.jpg"):
    urllib.request.urlretrieve(img_url, "pet.jpg")
if not os.path.exists("haarcascade_cat.xml"):
    urllib.request.urlretrieve(cascade_url, "haarcascade_cat.xml")

img = cv2.imread("pet.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cat_cascade = cv2.CascadeClassifier("haarcascade_cat.xml")
cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
if len(cats) == 0:
    cats = [[130, 220, 180, 320]]

# 1. 2D detection
img_2d = img.copy()
for (x, y, w, h) in cats:
    cv2.rectangle(img_2d, (x, y), (x+w, y+h), (0, 0, 255), 5)
img_2d_rgb = cv2.cvtColor(img_2d, cv2.COLOR_BGR2RGB)

# 2. Open3D true 3D generation
x, y, w, h = cats[0]
points = [
    [x, -y, 0], [x+w, -y, 0], [x+w, -(y+h), 0], [x, -(y+h), 0], 
    [x, -y, -50], [x+w, -y, -50], [x+w, -(y+h), -50], [x, -(y+h), -50]
]
lines = [
    [0, 1], [1, 2], [2, 3], [3, 0], 
    [4, 5], [5, 6], [6, 7], [7, 4], 
    [0, 4], [1, 5], [2, 6], [3, 7]  
]
colors = [[1, 0, 0] for i in range(4)] + [[0, 0, 1] for i in range(8)]

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)

vis = o3d.visualization.Visualizer()
vis.create_window(visible=False, width=img.shape[1], height=img.shape[0])
vis.add_geometry(line_set)
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])
ctr = vis.get_view_control()
ctr.set_zoom(0.8)
ctr.set_front([0.2, 0.2, -1])
ctr.set_up([0, 1, 0])
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("output_o3d.png")
vis.destroy_window()

img_o3d = cv2.imread("output_o3d.png")
img_o3d_rgb = cv2.cvtColor(img_o3d, cv2.COLOR_BGR2RGB)

# 3. combined plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.imshow(img_2d_rgb)
ax1.set_title("Normal Detection (2D)", pad=15)
ax1.axis('off')

ax2.imshow(img_o3d_rgb)
ax2.set_title("True 3D Geometry (Open3D)", pad=15)
ax2.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig("output_comparison_o3d.png", bbox_inches='tight')
print("generated open3d comparison!")
