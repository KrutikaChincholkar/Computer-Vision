import cv2
import numpy as np
from google.colab.patches import cv2_imshow  # For displaying images in Google Colab

# Step 1: Load the image in grayscale
# cv2.IMREAD_GRAYSCALE converts the image to a single-channel grayscale image
img = cv2.imread("/content/DSC_0141-01.jpeg", cv2.IMREAD_GRAYSCALE)

# Step 2: Apply binary thresholding
# Pixels >127 become 255 (white), others become 0 (black)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Step 3: Find contours
# cv2.RETR_EXTERNAL retrieves only the outer contours
# cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments to save memory
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 4: Convert grayscale to color (for colored contour drawing)
imag_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Step 5: Draw the contours on the image
# -1 → draw all contours
# (0, 255, 0) → green color
# 2 → thickness of contour lines
cv2.drawContours(imag_contours, contours, -1, (0, 255, 0), 2)

# Step 6: Display results
cv2_imshow(img)              # Original grayscale image
cv2_imshow(binary)           # Binary (thresholded) image
cv2_imshow(imag_contours)    # Image with drawn contours
