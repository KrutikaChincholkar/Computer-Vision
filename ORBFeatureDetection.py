# Step 1: Import Libraries
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Step 2: Load the image
image = cv2.imread("/content/Waffle.jpg")  # Load input image
cv2_imshow(image)  # Display the original image

# Step 3: Convert the image to grayscale (required for ORB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 4: Initialize ORB detector
# ORB combines FAST (keypoint detection) and BRIEF (descriptor computation)
orb = cv2.ORB_create(nfeatures=500)  # You can adjust number of keypoints (default = 500)

# Step 5: Detect keypoints and compute descriptors
keypoints, descriptors = orb.detectAndCompute(gray, None)

# Step 6: Draw detected keypoints
orb_image = cv2.drawKeypoints(
    image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Step 7: Display results
cv2_imshow(orb_image)

# Step 8: Print key information
print("Number of Keypoints Detected:", len(keypoints))
print("Descriptor Shape:", descriptors.shape)
