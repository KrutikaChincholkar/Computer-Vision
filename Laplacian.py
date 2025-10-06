import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image
image = cv2.imread('/content/Screenshot 2024-03-19 144347.png')

# Convert to grayscale for edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Apply Laplacian filter
laplacian = cv2.Laplacian(gray, cv2.CV_64F)   # CV_64F -> 64-bit float precision
laplacian = cv2.convertScaleAbs(laplacian)    # Convert back to 8-bit

# Step 3: Display results
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian Edge Detection')
plt.axis('off')

plt.show()
