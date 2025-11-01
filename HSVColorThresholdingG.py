# Import necessary libraries
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Step 1: Load the image
image = cv2.imread("image.jpg")

# Step 2: Convert from BGR to HSV color space
# HSV makes color detection easier because hue corresponds directly to color.
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Step 3: Define range for green color
# Hue for green is approximately between 35–85 degrees.
lower_green = np.array([35, 50, 50])    # lower limit for green
upper_green = np.array([85, 255, 255])  # upper limit for green

# Step 4: Create a binary mask for green regions
mask = cv2.inRange(hsv, lower_green, upper_green)

# Step 5: Apply the mask to the original image
# This keeps only the green areas while setting other parts to black.
result = cv2.bitwise_and(image, image, mask=mask)

# Step 6: Display the results
cv2_imshow(image)   # Original image
cv2_imshow(mask)    # Binary mask (white = detected green areas)
cv2_imshow(result)  # Extracted green regions
