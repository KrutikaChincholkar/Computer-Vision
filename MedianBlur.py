import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load the image in grayscale
image = cv2.imread('/content/minion.png', 0)
cv2_imshow(image)

# Apply Median Blur
blurred = cv2.medianBlur(image, 5)
cv2_imshow(blurred)
