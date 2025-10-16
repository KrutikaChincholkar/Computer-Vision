import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Step 1: Read the image in grayscale mode
image = cv2.imread('/content/Screenshot 2024-03-19 144347.png', 0)

# Step 2: Apply Regular Histogram Equalization
hist_eq = cv2.equalizeHist(image)

# Step 3: Apply Adaptive Histogram Equalization (CLAHE)
# clipLimit = threshold for contrast limiting
# tileGridSize = size of the small regions (tiles)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_image = clahe.apply(image)

# Step 4: Display all three images for comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(hist_eq, cmap='gray')
plt.title('Histogram Equalized Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(clahe_image, cmap='gray')
plt.title('CLAHE (Adaptive Equalized) Image')
plt.axis('off')

plt.show()

# Optional: Display images using cv2_imshow (for Colab preview)
print("Original Image:")
cv2_imshow(image)
print("Histogram Equalized Image:")
cv2_imshow(hist_eq)
print("CLAHE (Adaptive Equalized) Image:")
cv2_imshow(clahe_image)
