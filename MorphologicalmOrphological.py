import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Step 1: Load the image in grayscale
img = cv2.imread("img.jpeg", cv2.IMREAD_GRAYSCALE)

# Step 2: Apply binary thresholding
# Any pixel value > 127 becomes 255 (white), else 0 (black)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Display the original grayscale image
cv2_imshow(img)

# Step 3: Define a kernel (5x5 matrix of ones)
# This defines the neighborhood for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Step 4: Perform Erosion
# Erosion shrinks white (foreground) areas — removes small white noise and detach objects
erosion = cv2.erode(binary, kernel, iterations=1)

# Step 5: Perform Dilation
# Dilation enlarges white regions — fills small black holes and connects nearby white objects
dilation = cv2.dilate(binary, kernel, iterations=1)

# Step 6: Display results
cv2_imshow(dilation)
cv2_imshow(erosion)
