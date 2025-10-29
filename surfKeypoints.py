# Step 1: Install opencv-contrib if not already installed
# (Uncomment this line and run once in Colab)
# !pip install opencv-contrib-python

# Step 2: Import Libraries
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Step 3: Load the image
image = cv2.imread("/content/Waffle.jpg")  # Read the input image
cv2_imshow(image)  # Display original image

# Step 4: Convert to grayscale (SURF works better on grayscale images)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 5: Initialize the SURF detector
hessian_threshold = 400  # Controls number of keypoints (higher = fewer)
surf = cv2.xfeatures2d.SURF_create(hessian_threshold)

# Step 6: Detect keypoints and compute descriptors
keypoints, descriptors = surf.detectAndCompute(gray, None)
print("Number of keypoints detected:", len(keypoints))
print("Descriptor shape:", descriptors.shape)

# Step 7: Draw keypoints on the image
# DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS shows scale and orientation
surf_image = cv2.drawKeypoints(image, keypoints, None,
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Step 8: Display the image with keypoints
cv2_imshow(surf_image)
