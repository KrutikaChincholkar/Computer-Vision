# Import required libraries
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Step 1: Load the input image
image = cv2.imread("image.jpg")

# Step 2: Convert from BGR (default in OpenCV) to HSV color space
# HSV = Hue (color), Saturation (intensity), Value (brightness)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Step 3: Define the range for red color in HSV
# Red appears in two regions of the hue circle (0–10 and 170–180)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Step 4: Create two masks and combine them (to capture both red ranges)
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2

# Step 5: Apply mask to extract only red regions
result = cv2.bitwise_and(image, image, mask=mask)

# Step 6: Display results
cv2_imshow(image)   # Original image
cv2_imshow(mask)    # Binary mask (white = red areas)
cv2_imshow(result)  # Extracted red color regions
