import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load the image
image = cv2.imread('img.png')

# Show original image
cv2_imshow(image)

# Define rotation parameters
angle = -45  # Negative for clockwise rotation
(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# Get rotation matrix
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

# Compute new bounding dimensions to avoid cropping
cos = np.abs(rotation_matrix[0, 0])
sin = np.abs(rotation_matrix[0, 1])

new_w = int((h * sin) + (w * cos))
new_h = int((h * cos) + (w * sin))

# Adjust rotation matrix to consider translation
rotation_matrix[0, 2] += (new_w / 2) - center[0]
rotation_matrix[1, 2] += (new_h / 2) - center[1]

# Perform the rotation
rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

# Show the rotated image
cv2_imshow(rotated)
