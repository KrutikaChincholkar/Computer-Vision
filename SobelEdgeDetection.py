import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('img_path.png', cv2.IMREAD_GRAYSCALE)

# Apply Sobel filters
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

# Combine results (gradient magnitude)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Convert to 8-bit image for visualization
sobel_combined = cv2.convertScaleAbs(sobel_combined)

# Display the images
cv2_imshow(image)
cv2_imshow(sobel_combined)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(np.abs(sobel_x), cmap='gray')
plt.title('Sobel X (Horizontal)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(sobel_y), cmap='gray')
plt.title('Sobel Y (Vertical)')
plt.axis('off')

plt.show()
