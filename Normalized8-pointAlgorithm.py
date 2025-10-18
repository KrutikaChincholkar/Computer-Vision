import cv2
import numpy as np

# Define corresponding points in two images
# (Usually obtained from feature matching like SIFT, SURF, or ORB)
pts1 = np.array([
    [100, 200],
    [120, 220],
    [130, 250],
    [150, 270],
    [170, 290],
    [200, 310],
    [220, 330],
    [250, 350]
])

pts2 = np.array([
    [105, 210],
    [125, 230],
    [140, 255],
    [160, 280],
    [180, 300],
    [220, 330],
    [230, 370],
    [270, 420]
])

# Compute the Fundamental Matrix using the 8-point algorithm
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

# Check and display the result
if F is not None:
    print("✅ Fundamental Matrix found:\n")
    print(F)
else:
    print("❌ Fundamental Matrix could not be computed.")
