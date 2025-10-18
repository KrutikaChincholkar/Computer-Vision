import numpy as np
import cv2

# Define two 3D points (for demonstration)
p1 = np.array([0, 0, 0, 1])   # Camera 1 origin
p2 = np.array([1, 0, 0, 1])   # Camera 2 shifted 1 unit on X-axis

# Define Projection Matrices for two cameras
# P = K [R | t], simplified here as Identity & translation
P1 = np.array([[1, 0, 0, 0],   # Camera 1 at origin
               [0, 1, 0, 0],
               [0, 0, 1, 0]], dtype=np.float32)

P2 = np.array([[1, 0, 0, -1],  # Camera 2 shifted along X by -1
               [0, 1, 0, 0],
               [0, 0, 1, 0]], dtype=np.float32)

# Image points in homogeneous coordinates (2D points from each image)
x1 = np.array([[150], [200]], dtype=np.float32)
x2 = np.array([[100], [150]], dtype=np.float32)

# Triangulate the 3D point from the two views
x_homogeneous = cv2.triangulatePoints(P1, P2, x1, x2)
print("Homogeneous coordinates of the 3D point:\n", x_homogeneous)

# Convert from homogeneous to Cartesian coordinates (divide by w)
x = x_homogeneous / x_homogeneous[3]
print("\n3D coordinates after normalization:\n", x)
