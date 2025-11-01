import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Step 1: Load the image
image = cv2.imread("/content/OIP.jpg")

# Step 2: Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Step 3: Define the HSV range for yellow color
# Hue for yellow lies approximately between 20°–30°.
# S (Saturation) and V (Value) control vividness and brightness.
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Step 4: Create a binary mask where yellow areas appear white
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Step 5: Apply mask on the original image to isolate yellow regions
result = cv2.bitwise_and(image, image, mask=mask)

# Step 6: Display original and result images
cv2_imshow(image)   # Original image
cv2_imshow(mask)    # (Optional) Binary mask
cv2_imshow(result)  # Output showing only yellow parts
