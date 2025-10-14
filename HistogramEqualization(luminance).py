import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Load the image (in BGR format)
image = cv2.imread('/content/Screenshot 2024-10-18 105704.png')

# Convert to YUV color space
image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

# Equalize only the Y (luminance) channel
image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])

# Convert back to BGR color space
equalized_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

# Plot histograms for original and equalized images
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray')
plt.title('Original Image Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(equalized_image.ravel(), bins=256, range=(0, 256), color='gray')
plt.title('Equalized Image Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

# Convert BGR → RGB for correct color display in matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
equalized_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)

# Display Original and Equalized Images
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_rgb)
plt.title('Equalized Image')
plt.axis('off')

plt.show()
