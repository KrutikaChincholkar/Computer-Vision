import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Step 1: Load the image
img = cv2.imread("img.jpeg")

# Step 2: Convert the image to the Lab color space
# LAB separates luminance (L) and color components (A and B),
# which helps achieve better color segmentation than RGB.
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Step 3: Apply Mean Shift segmentation
# sp = spatial window radius (in pixels)
# sr = color window radius (in Lab color space)
# The algorithm groups pixels based on color and spatial proximity.
segmented_img = cv2.pyrMeanShiftFiltering(img_lab, sp=21, sr=51)

# Step 4: Convert the result back to BGR for visualization
segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_LAB2BGR)

# Step 5: Display the original and segmented images
cv2_imshow(img)
cv2_imshow(segmented_img)
