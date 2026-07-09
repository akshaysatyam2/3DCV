import os
import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt

# download assets
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
h_img, w_img = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cascade = cv2.CascadeClassifier("haarcascade_cat.xml")
cats = cascade.detectMultiScale(gray, 1.1, 3)

if len(cats) == 0:
    cats = [[130, 220, 180, 320]]

x, y, w, h = cats[0]

# 1. 2D OpenCV bounding box
img_2d = img.copy()
cv2.rectangle(img_2d, (x, y), (x+w, y+h), (0, 0, 255), 5)
img_2d_rgb = cv2.cvtColor(img_2d, cv2.COLOR_BGR2RGB)

# 2. OpenCV pseudo-3D projection (vanishing point trick)
img_3d = img.copy()
img_cx, img_cy = w_img / 2, h_img / 2
scale = 0.8

pts_front = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], np.int32)
pts_back = np.array(
    [[int(img_cx + (px - img_cx) * scale),
      int(img_cy + (py - img_cy) * scale)]
     for px, py in pts_front], np.int32
)

cv2.polylines(img_3d, [pts_back], True, (255, 0, 0), 3)
for i in range(4):
    cv2.line(img_3d, tuple(pts_front[i]), tuple(pts_back[i]), (255, 0, 0), 2)
cv2.polylines(img_3d, [pts_front], True, (0, 0, 255), 3)
img_3d_rgb = cv2.cvtColor(img_3d, cv2.COLOR_BGR2RGB)

# 3. True 3D projection using cv2.projectPoints
img_cv3d = img.copy()

cx_box = x + w / 2
cy_box = y + h / 2
box_d = max(w, h) * 0.6

half_w = w / 2
half_h = h / 2
half_d = box_d / 2

box_3d = np.array([
    [-half_w, -half_h, -half_d],
    [ half_w, -half_h, -half_d],
    [ half_w,  half_h, -half_d],
    [-half_w,  half_h, -half_d],
    [-half_w, -half_h,  half_d],
    [ half_w, -half_h,  half_d],
    [ half_w,  half_h,  half_d],
    [-half_w,  half_h,  half_d],
], dtype=np.float32)

f = float(w_img)
K = np.array([[f, 0, w_img / 2],
              [0, f, h_img / 2],
              [0, 0, 1]], dtype=np.float64)

tx = (cx_box - w_img / 2)
ty = (cy_box - h_img / 2)

rvec = np.array([0.0, 0.3, 0.0])
tvec = np.array([tx, ty, f * 1.2], dtype=np.float64)
dist_coeffs = np.zeros(4)

projected, _ = cv2.projectPoints(box_3d, rvec, tvec, K, dist_coeffs)
projected = projected.reshape(-1, 2).astype(int)

edges_front = [(0,1),(1,2),(2,3),(3,0)]
edges_back  = [(4,5),(5,6),(6,7),(7,4)]
edges_side  = [(0,4),(1,5),(2,6),(3,7)]

# back face and connections in blue (BGR: 255, 0, 0)
for i, j in edges_back:
    cv2.line(img_cv3d, tuple(projected[i]), tuple(projected[j]), (255, 0, 0), 3)
for i, j in edges_side:
    cv2.line(img_cv3d, tuple(projected[i]), tuple(projected[j]), (255, 0, 0), 2)
# front face in red (BGR: 0, 0, 255)
for i, j in edges_front:
    cv2.line(img_cv3d, tuple(projected[i]), tuple(projected[j]), (0, 0, 255), 3)

img_cv3d_rgb = cv2.cvtColor(img_cv3d, cv2.COLOR_BGR2RGB)

# 4. Final Comparison
fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131)
ax1.imshow(img_2d_rgb)
ax1.set_title("2D OpenCV Box", pad=15)
ax1.axis('off')

ax2 = fig.add_subplot(132)
ax2.imshow(img_3d_rgb)
ax2.set_title("Pseudo-3D Projection (Vanishing Trick)", pad=15)
ax2.axis('off')

ax3 = fig.add_subplot(133)
ax3.imshow(img_cv3d_rgb)
ax3.set_title("True 3D Projection (cv2.projectPoints)", pad=15)
ax3.axis('off')

plt.tight_layout()
plt.savefig("output_comparison.png", bbox_inches='tight')
print("generated perfect cv2 projectpoints comparison!")
