import numpy as np

# Step 1: Define the Homography matrix (3x3)
# This matrix represents a combination of translation, rotation, scaling, and perspective distortion.
H = np.array([
    [1, 0.5, 10],   # affects x-axis
    [0.2, 1, 5],    # affects y-axis
    [0.01, 0.02, 1] # affects perspective
])

# Step 2: Define a 2D point in homogeneous coordinates
# A 2D point (x, y) becomes [x, y, 1] for matrix multiplication
point = np.array([2, 3, 1])

# Step 3: Apply the homography transformation
# Perform matrix multiplication (H × point)
trans_point = np.dot(H, point)

# Step 4: Normalize the result
# Divide by the last element to convert back from homogeneous coordinates
trans_point /= trans_point[-1]

# Step 5: Print the transformed coordinates
print("Original point: ", point[:-1])
print("Transformed coordinates: ", trans_point[:-1])
