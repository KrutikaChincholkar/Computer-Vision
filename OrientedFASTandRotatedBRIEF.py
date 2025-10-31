# Import required libraries
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Step 1: Load two grayscale images
img1 = cv2.imread("Image.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("Image.jpg", cv2.IMREAD_GRAYSCALE)

# Step 2: Initialize the ORB detector
orb = cv2.ORB_create(nfeatures=1000)  # You can adjust number of features if needed

# Step 3: Detect keypoints and compute descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

print("Number of Keypoints in Image 1:", len(kp1))
print("Number of Keypoints in Image 2:", len(kp2))

# Step 4: Create a Brute Force Matcher with Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Step 5: Match descriptors between the two images
matches = bf.match(des1, des2)

# Step 6: Sort matches based on descriptor distance (lower = better match)
matches = sorted(matches, key=lambda x: x.distance)

# Step 7: Draw the top N matches (100 in this case)
img3 = cv2.drawMatches(
    img1, kp1, img2, kp2, matches[:100], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Step 8: Display the matched keypoints
cv2_imshow(img3)
