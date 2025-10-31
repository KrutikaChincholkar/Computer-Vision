# Step 1: Import necessary libraries
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Step 2: Load the image
image = cv2.imread("/content/Waffle.jpg")  # Load the image from path
cv2_imshow(image)  # Display the original image

# Step 3: Convert image to grayscale (required for feature detection)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 4: Initialize FAST feature detector (for keypoint detection)
fast = cv2.FastFeatureDetector_create()

# Step 5: Detect keypoints using FAST
keypoints = fast.detect(gray, None)

# Step 6: Initialize BRISK descriptor extractor (for feature description)
brisk = cv2.BRISK_create()

# Step 7: Compute descriptors for the detected FAST keypoints
keypoints, descriptors = brisk.compute(gray, keypoints)

# Step 8: Draw detected keypoints (visualization)
brisk_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Step 9: Display images
cv2_imshow(brisk_image)

# Step 10: Print details
print("Number of keypoints detected:", len(keypoints))
print("Descriptor shape:", descriptors.shape)
