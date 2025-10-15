import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Define gamma correction function
def gamma_correction(image, gamma):
    # Build a lookup table mapping pixel values [0,255] -> adjusted values
    gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, gamma_table)

# Read the image (grayscale)
image = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)

# Apply different gamma corrections
gamma_low = gamma_correction(image, 0.5)   # Brightens the image
gamma_high = gamma_correction(image, 2.0)  # Darkens the image

# Display using OpenCV
cv2_imshow(image)
cv2_imshow(gamma_low)
cv2_imshow(gamma_high)

# OR (better visualization in Colab) using matplotlib
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gamma_low, cmap='gray')
plt.title('Gamma 0.5 (Brighter)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gamma_high, cmap='gray')
plt.title('Gamma 2.0 (Darker)')
plt.axis('off')

plt.show()
