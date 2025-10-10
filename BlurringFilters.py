import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load grayscale image
image = cv2.imread('/content/minion.png', 0)
cv2_imshow(image)

# Apply bilateral filter
blurred = cv2.bilateralFilter(image, 9, 75, 75)
cv2_imshow(blurred)
