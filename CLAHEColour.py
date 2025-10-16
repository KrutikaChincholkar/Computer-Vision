import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Load the image in color
image = cv2.imread('img.png')

# Convert the image to LAB color space
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Split into L (lightness), A, and B channels
l_channel, a_channel, b_channel = cv2.split(lab_image)

# Apply CLAHE to the L channel only (for contrast enhancement)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_l_channel = clahe.apply(l_channel)

# Merge enhanced L with original A and B channels
enhanced_lab_image = cv2.merge((enhanced_l_channel, a_channel, b_channel))

# Convert back to BGR color space
enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

# Display images using cv2_imshow (for Google Colab)
print("Original Image:")
cv2_imshow(image)
print("Enhanced Image (After CLAHE):")
cv2_imshow(enhanced_image)

# Display side-by-side with matplotlib
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
plt.title("CLAHE Enhanced Image")
plt.axis("off")

plt.show()
