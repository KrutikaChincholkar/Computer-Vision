import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image in grayscale mode
image = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate minimum and maximum pixel intensities
min_intensity = np.min(image)
max_intensity = np.max(image)

print("Minimum Intensity:", min_intensity)
print("Maximum Intensity:", max_intensity)

# Perform contrast stretching
stretched_image = ((image - min_intensity) / (max_intensity - min_intensity)) * 255

# Convert to unsigned 8-bit integer
stretched_image = stretched_image.astype(np.uint8)

# Display Original and Stretched Images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(stretched_image, cmap='gray')
plt.title('Contrast Stretched Image')
plt.axis("off")

plt.show()

# Optionally, compare histograms
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray')
plt.title('Original Histogram')

plt.subplot(1, 2, 2)
plt.hist(stretched_image.ravel(), bins=256, range=(0, 256), color='gray')
plt.title('Stretched Histogram')
plt.show()
