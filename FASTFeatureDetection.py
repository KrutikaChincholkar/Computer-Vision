#FAST (Features from Accelerated Segment Test) feature detection
# Step 1: Import necessary libraries
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Step 2: Load the image
image = cv2.imread("/content/Waffle.jpg")  # Read the image
cv2_imshow(image)  # Display the original image

# Step 3: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

# Step 4: Initialize the FAST feature detector
fast = cv2.FastFeatureDetector_create()  # Create a FAST detector object

# Step 5: Detect keypoints using FAST
keypoints = fast.detect(gray, None)  # Detect corners/features

# Step 6: Draw keypoints on the original image
fast_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

# Step 7: Display results
cv2_imshow(fast_image)
print("Number of keypoints detected:", len(keypoints))
