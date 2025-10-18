import numpy as np
import cv2

# Stereo camera parameters
f = 700     # Focal length in pixels (typical for stereo cameras)
B = 0.2     # Baseline distance between the two cameras in meters

# Pixel coordinates of the same point in left and right images
xL = 120    # x-coordinate in left image
xR = 110    # x-coordinate in right image

# Step 1: Compute disparity
disparity = xL - xR   # difference in horizontal positions
print("Disparity:", disparity)

# Step 2: Compute depth using stereo geometry
# Formula: Depth (Z) = (f * B) / disparity
depth = (f * B) / disparity
print("Estimated Depth (Z):", depth, "meters")
