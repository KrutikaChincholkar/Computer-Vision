import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Define gamma correction function
def gamma_correction(image, gamma):
    # Build lookup table for gamma correction
    gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    # Apply gamma correction using lookup table
    return cv2.LUT(image, gamma_table)

# Read the image in color
image = cv2.imread('/content/Screenshot 2024-03-19 144347.png')

# Apply gamma correction
gamma_low = gamma_correction(image, 0.5)   # Brighter
gamma_high = gamma_correction(image, 5.0)  # Darker

# Display using OpenCV (Colab friendly)
cv2_imshow(image)
cv2_imshow(gamma_low)
cv2_imshow(gamma_high)

# OR visualize neatly using matplotlib
# Convert BGR → RGB for correct color display in matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gamma_low_rgb = cv2.cvtColor(gamma_low, cv2.COLOR_BGR2RGB)
gamma_high_rgb = cv2.cvtColor(gamma_high, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gamma_low_rgb)
plt.title('Gamma 0.5 (Brighter)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gamma_high_rgb)
plt.title('Gamma 5.0 (Darker)')
plt.axis('off')

plt.show()
