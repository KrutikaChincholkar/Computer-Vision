import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load the image
image = cv2.imread('/content/Screenshot 2024-03-19 134750.png')
cv2_imshow(image)

# Define scaling factors
sx, sy = 1.5, 1.5  # 1.5x larger in both directions

# Scale (resize) the image
scaled = cv2.resize(image, None, fx=sx, fy=sy, interpolation=cv2.INTER_CUBIC)

# Display scaled image
cv2_imshow(scaled)
