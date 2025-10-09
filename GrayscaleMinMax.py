import cv2
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from google.colab.patches import cv2_imshow

# Load grayscale image
image = cv2.imread('img_path.png', cv2.IMREAD_GRAYSCALE)
cv2_imshow(image)

# Define kernel size
ks = 3

# Apply maximum and minimum filters
max_filter = maximum_filter(image, size=ks)
min_filter = minimum_filter(image, size=ks)

# Display results
cv2_imshow(max_filter)
cv2_imshow(min_filter)
