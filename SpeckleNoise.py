import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Function to generate speckle noise
def generate_speckle_noise(image_shape, mean=0, variance=0.01):
    noise = np.random.normal(mean, np.sqrt(variance), size=image_shape)
    speckle_noise = noise * np.ones(image_shape)
    return speckle_noise

# Generate a blank grayscale image (black)
image_shape = (512, 512)
image = np.zeros(image_shape, dtype=np.float32)

# Add speckle noise
speckle_variance = 0.01
speckle_noise = generate_speckle_noise(image_shape, variance=speckle_variance)
noisy_image = image + speckle_noise

# Clip to valid range [0, 1]
noisy_image = np.clip(noisy_image, 0, 1)

# Display the noisy image
plt.figure(figsize=(6, 6))
plt.imshow(noisy_image, cmap='gray')
plt.title('Image with Speckle Noise')
plt.axis('off')
plt.show()

img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
speckle_noise = generate_speckle_noise(img.shape, variance=0.02)
noisy_img = img + img * speckle_noise
noisy_img = np.clip(noisy_img, 0, 1)

plt.imshow(noisy_img, cmap='gray')
plt.title('Real Image with Speckle Noise')
plt.axis('off')
plt.show()
