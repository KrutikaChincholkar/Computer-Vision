import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Step 1: Read the input image
image = cv2.imread('img.png')

# Step 2: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2_imshow(gray)

# Step 3: Convert image to float32 (required by cornerHarris)
gray = np.float32(gray)

# Step 4: Set Harris Corner Detector parameters
block_size = 2   # Neighborhood size (the size of the window considered around each pixel)
ksize = 3        # Aperture parameter for the Sobel() operator (derivative size)
k = 0.04         # Harris detector free parameter (typically between 0.04 and 0.06)

# Step 5: Detect corners using Harris algorithm
dst = cv2.cornerHarris(gray, block_size, ksize, k)

# Step 6: Dilate the corner points to enhance visibility
dst = cv2.dilate(dst, None)

# Step 7: Threshold for an optimal value and mark corners in red
image[dst > 0.01 * dst.max()] = [0, 0, 255]

# Step 8: Display the final image with corners marked
cv2_imshow(image)
