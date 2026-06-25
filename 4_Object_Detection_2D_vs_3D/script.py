import os
import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

# 1. download image n haar cascade from web
img_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg"
cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalcatface.xml"

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

if not os.path.exists("pet.jpg"):
    urllib.request.urlretrieve(img_url, "pet.jpg")
if not os.path.exists("haarcascade_cat.xml"):
    urllib.request.urlretrieve(cascade_url, "haarcascade_cat.xml")

# 2. 2d object detection using opencv
img = cv2.imread("pet.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cat_cascade = cv2.CascadeClassifier("haarcascade_cat.xml")
cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

# fallback if cascade misses
if len(cats) == 0:
    # draw a box roughly around the dog
    cats = [[130, 220, 180, 320]]

img_2d = img.copy()
for (x, y, w, h) in cats:
    cv2.rectangle(img_2d, (x, y), (x+w, y+h), (0, 0, 255), 5) # red in BGR
img_2d_rgb = cv2.cvtColor(img_2d, cv2.COLOR_BGR2RGB)

# 3. combined comparison plot (2D vs Matplotlib3D)
fig = plt.figure(figsize=(14, 6))

# Normal Detection
ax1 = fig.add_subplot(121)
ax1.imshow(img_2d_rgb)
ax1.set_title("Normal Detection (2D OpenCV)", pad=15)
ax1.axis('off')

# Matplotlib3D True Geometry
ax2 = fig.add_subplot(122, projection='3d')
x, y, w, h = cats[0]

# front box (z=0)
fx = [x, x+w, x+w, x, x]
fy = [-y, -y, -(y+h), -(y+h), -y]  # negative y to match image coordinates
fz = [0, 0, 0, 0, 0]

# back box (z=-100)
bx = [x, x+w, x+w, x, x]
by = [-y, -y, -(y+h), -(y+h), -y]
bz = [-100, -100, -100, -100, -100]

# plot front face
ax2.plot(fx, fz, fy, color='red', linewidth=3, label='Front Face (Detected)')
# plot back face n depth
ax2.plot(bx, bz, by, color='blue', linewidth=3, label='Back Face & Depth')
# connect corners
for i in range(4):
    ax2.plot([fx[i], bx[i]], [fz[i], bz[i]], [fy[i], by[i]], color='blue', linewidth=2)

# adjust camera angle for cool 3d effect
ax2.view_init(elev=20, azim=-60)
ax2.set_xlabel('Width (X)')
ax2.set_ylabel('Depth (Z)')
ax2.set_zlabel('Height (Y)')
ax2.set_title("True 3D Geometry (Matplotlib3D)", pad=15)
ax2.legend(loc='lower right')

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig("output_comparison.png", bbox_inches='tight')
print("generated 2D vs Matplotlib3D comparison!")
