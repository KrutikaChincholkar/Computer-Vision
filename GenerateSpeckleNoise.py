import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Function to generate speckle noise
def generate_speckle_noise(image_shape, mean=0, variance=0.01):
    # Generate Gaussian noise
    noise = np.random.normal(mean, np.sqrt(variance), size=image_shape)
    # Scale the noise
    speckle_noise = noise * np.ones(image_shape)
    return speckle_noise

# Read the input image (color)
image = cv2.imread('/content/click1.jpg')
cv2_imshow(image)

# Generate speckle noise and add it to the image
speckle_variance = 0.01
speckle_noise = generate_speckle_noise(image.shape, variance=speckle_variance)

# Add speckle noise (multiplicative noise)
noisy_image = image + image * speckle_noise

# Clip pixel values to valid range and convert to uint8
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Display noisy image using OpenCV
cv2_imshow(noisy_image)

# Display using matplotlib (better for visualization)
plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
plt.title('Image with Speckle Noise')
plt.axis('off')
plt.show()

# Optionally, save the noisy image
cv2.imwrite('speckle_noise_image.png', noisy_image)
print("Speckle noise image saved as 'speckle_noise_image.png'")
