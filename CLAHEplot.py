import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Load the image in grayscale mode
image = cv2.imread('/content/Screenshot 2024-03-19 144347.png', 0)

# Create a CLAHE object (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

# Apply CLAHE to the grayscale image
clahe_image = clahe.apply(image)

# Display the original and CLAHE images
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(clahe_image, cmap='gray')
plt.title('CLAHE Enhanced Image')
plt.axis('off')

plt.show()

# Optional: Show with cv2_imshow (for Colab display)
print("Original Image:")
cv2_imshow(image)
print("CLAHE Enhanced Image:")
cv2_imshow(clahe_image)
