# (Scale-Invariant Feature Transform)
# Step 1: Import required libraries
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Step 2: Load the image
image = cv2.imread("/content/IMG_20210111_115206_341.jpg")  # Read the image
cv2_imshow(image)  # Display the original image

# Step 3: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert BGR → Grayscale

# Step 4: Initialize the SIFT detector
sift = cv2.SIFT_create()  # Create a SIFT detector object

# Step 5: Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

print("Number of Keypoints detected:", len(keypoints))
print("Shape of Descriptor Matrix:", descriptors.shape)

# Step 6: Draw keypoints on the original image
# Flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS draws circles and orientation
sift_image = cv2.drawKeypoints(image, keypoints, None,
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Step 7: Display the image with keypoints
cv2_imshow(sift_image)
