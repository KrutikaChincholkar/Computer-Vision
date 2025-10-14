import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Step 1: Define the speckle noise generator function
def generate_speckle_noise(image_shape, mean=0, variance=0.01):
    """Generates multiplicative speckle noise"""
    noise = np.random.normal(mean, np.sqrt(variance), size=image_shape)
    return noise

# Step 2: Load the image
image = cv2.imread('/content/click1.jpg')
cv2_imshow(image)

# Step 3: Generate speckle noise and add it to the image
speckle_variance = 0.01
speckle_noise = generate_speckle_noise(image.shape, variance=speckle_variance)

# Speckle noise is multiplicative
noisy_image = image + image * speckle_noise

# Step 4: Clip pixel values and convert to uint8
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Step 5: Display the noisy image
cv2_imshow(noisy_image)

# Better visualization using Matplotlib (convert BGR → RGB)
plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
plt.title('Image with Speckle Noise')
plt.axis('off')
plt.show()

# Step 6: Save the noisy image
cv2.imwrite('speckle_noise_image.png', noisy_image)
print("✅ Speckle noise image saved as 'speckle_noise_image.png'")
