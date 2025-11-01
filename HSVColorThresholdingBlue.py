# Import required libraries
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Step 1: Load the image
image = cv2.imread("/content/OIP.jpg")

# Step 2: Convert the image from BGR to HSV color space
# HSV (Hue, Saturation, Value) separates color (hue) from intensity (value), 
# making color-based detection easier and more robust.
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Step 3: Define the range for the blue color in HSV
# Hue values for blue lie approximately between 100° and 130°.
# Saturation and Value control how vivid and bright the color is.
lower_blue = np.array([100, 50, 50])    # Lower threshold
upper_blue = np.array([130, 255, 255])  # Upper threshold

# Step 4: Create a binary mask where blue colors are white and others are black
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Step 5: Apply the mask to the original image to extract blue regions
# This operation keeps only the parts of the image that fall in the blue range.
result = cv2.bitwise_and(image, image, mask=mask)

# Step 6: Display the images
cv2_imshow(image)   # Original image
cv2_imshow(mask)    # Binary mask (white = detected blue areas)
cv2_imshow(result)  # Final image showing only blue regions
