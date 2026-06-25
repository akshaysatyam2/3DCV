import os
import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt

# 1. download image n haar cascade from web
img_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

urllib.request.urlretrieve(img_url, "lena.jpg")
urllib.request.urlretrieve(cascade_url, "haarcascade.xml")

# 2. 2d object detection using opencv
img = cv2.imread("lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier("haarcascade.xml")
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

img_2d = img.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(img_2d, (x, y), (x+w, y+h), (255, 0, 0), 5)

# 3. simulated 3d detection (drawing a 3d box)
img_3d = img.copy()
for (x, y, w, h) in faces:
    # 2d front points
    pts_front = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], np.int32)
    
    # simulated 3d back points with depth offset
    offset_x = int(w * 0.25)
    offset_y = int(h * 0.25)
    pts_back = np.array([[x+offset_x, y-offset_y], [x+w+offset_x, y-offset_y], 
                         [x+w+offset_x, y+h-offset_y], [x+offset_x, y+h-offset_y]], np.int32)
    
    # draw back box (green)
    cv2.polylines(img_3d, [pts_back], True, (0, 255, 0), 3)
    # draw front box (red)
    cv2.polylines(img_3d, [pts_front], True, (0, 0, 255), 3)
    # connect front n back corners (blue)
    for i in range(4):
        cv2.line(img_3d, tuple(pts_front[i]), tuple(pts_back[i]), (255, 0, 0), 2)

# convert BGR to RGB for saving with matplotlib properly
img_2d_rgb = cv2.cvtColor(img_2d, cv2.COLOR_BGR2RGB)
img_3d_rgb = cv2.cvtColor(img_3d, cv2.COLOR_BGR2RGB)

# 4. save separate images
plt.imsave("output_2d.png", img_2d_rgb)
plt.imsave("output_3d.png", img_3d_rgb)

# 5. combined comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(img_2d_rgb)
ax1.set_title("2D Object Detection (Bounding Box)")
ax1.axis('off')

ax2.imshow(img_3d_rgb)
ax2.set_title("3D Object Detection (3D Bounding Box)")
ax2.axis('off')

plt.tight_layout()
plt.savefig("output_comparison.png")
print("generated 2d, 3d, n comparison images!")
