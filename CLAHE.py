import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Step 1: Load the original color image
image = cv2.imread('/content/Screenshot 2024-03-19 144347.png')

# Step 2: Convert the image from BGR to LAB color space
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Step 3: Split the LAB image into its three channels
l_channel, a_channel, b_channel = cv2.split(lab_image)

# Step 4: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on L-channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_l_channel = clahe.apply(l_channel)

# Step 5: Merge the enhanced L-channel back with the original A and B channels
enhanced_lab_image = cv2.merge((enhanced_l_channel, a_channel, b_channel))

# Step 6: Convert the LAB image back to BGR color space for display
enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

# Step 7: Display both the original and CLAHE-enhanced images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
plt.title('CLAHE Enhanced Image (LAB Space)')
plt.axis('off')

plt.show()

# Optional: Show in Colab (side-by-side)
print("Original Image:")
cv2_imshow(image)
print("CLAHE Enhanced Image:")
cv2_imshow(enhanced_image)
