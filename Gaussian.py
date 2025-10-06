# cv2.GaussianBlur(): Applies a Gaussian kernel for smoothing

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Step 1: Load the grayscale image
image = cv2.imread('/content/minion.png', cv2.IMREAD_GRAYSCALE)
cv2_imshow(image)
print("Original Image")

# Step 2: Add Gaussian noise
mean = 0
std_dev = 25
gaussian_noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)

# Add noise to the image safely (keep pixel range 0–255)
noisy_image = image.astype(np.float32) + gaussian_noise
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

cv2_imshow(noisy_image)
print("Image with Gaussian Noise")

# Step 3: Apply Gaussian Blur to reduce noise
filtered_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)

cv2_imshow(filtered_image)
print("After Gaussian Blur Filtering")
