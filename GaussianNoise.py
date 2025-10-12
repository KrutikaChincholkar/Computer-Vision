import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load the image in color mode
img = cv2.imread('img.png')
cv2_imshow(img)

# Load the image in grayscale mode
image = cv2.imread('img', cv2.IMREAD_GRAYSCALE)

# Generate Gaussian noise with mean = 0 and standard deviation = 25
mean = 0
std = 25
noise = np.random.normal(mean, std, image.shape).astype(np.float32)

# Convert image to float32 for precise addition
image_float = image.astype(np.float32)

# Add noise to the image
noisy_image = cv2.add(image_float, noise)

# Clip pixel values to [0, 255] and convert back to uint8
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Display original and noisy images
cv2_imshow(image)
cv2_imshow(noisy_image)

# Save the noisy image
cv2.imwrite("gaussian_noise.png", noisy_image)
print("Image saved as 'gaussian_noise.png'")
