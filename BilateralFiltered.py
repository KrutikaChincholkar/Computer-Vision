import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load the image (color image)
image = cv2.imread('/content/MVI_20011_img00001.jpg')

# Apply bilateral filter
# d = 9 → Diameter of each pixel neighborhood
# sigmaColor → Filter sigma in color space (higher = more color mixing)
# sigmaSpace → Filter sigma in coordinate space (higher = smoother)
bilateral_filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# Save the result
cv2.imwrite('/content/bilateral_filtered.jpg', bilateral_filtered)

# Display original and filtered images
print("Original Image:")
cv2_imshow(image)
print("Bilateral Filtered Image:")
cv2_imshow(bilateral_filtered)
