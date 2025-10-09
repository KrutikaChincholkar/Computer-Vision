import cv2
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from google.colab.patches import cv2_imshow

# Load the image (in color)
image = cv2.imread('/content/minion.png')
cv2_imshow(image)

# Define kernel size
ks = 3

# Apply maximum (dilation-like) filter
max_filter = maximum_filter(image, size=ks)

# Apply minimum (erosion-like) filter
min_filter = minimum_filter(image, size=ks)

# Display results
cv2_imshow(max_filter)
cv2_imshow(min_filter)
