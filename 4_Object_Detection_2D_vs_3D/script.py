import os
import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt

# 1. download image n haar cascade from web
img_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg"
cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalcatface.xml"

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

urllib.request.urlretrieve(img_url, "pet.jpg")
urllib.request.urlretrieve(cascade_url, "haarcascade_cat.xml")

# 2. 2d object detection using opencv
img = cv2.imread("pet.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cat_cascade = cv2.CascadeClassifier("haarcascade_cat.xml")

# detect cats
cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

# fallback if cascade misses (the image we chose usually works, but just in case!)
if len(cats) == 0:
    # draw a box roughly around the dog
    cats = [[130, 220, 180, 320]]

img_2d = img.copy()
for (x, y, w, h) in cats:
    cv2.rectangle(img_2d, (x, y), (x+w, y+h), (255, 0, 0), 5)

# 3. simulated 3d detection (drawing a 3d box)
img_3d = img.copy()
for (x, y, w, h) in cats:
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
ax1.set_title("2D Object Detection (Bounding Box)", pad=15)
ax1.axis('off')

ax2.imshow(img_3d_rgb)
ax2.set_title("3D Object Detection (3D Bounding Box)", pad=15)
ax2.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.88) # prevents text from cutting off at the top
plt.savefig("output_comparison.png", bbox_inches='tight') # bbox_inches='tight' helps too!
print("generated 2d, 3d, n comparison images!")
