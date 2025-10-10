# cv2.Canny(): Implements the Canny edge detection algorithm
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load image in grayscale
image = cv2.imread('/content/minion.png', 0)
cv2_imshow(image)

# Optional: Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2_imshow(blurred)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 100, 200)

# Display the edge-detected image
cv2_imshow(edges)
