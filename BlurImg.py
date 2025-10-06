import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Read the image from path
image = cv2.imread('/content/Screenshot 2024-03-19 144347.png')

# Check if the image was loaded correctly
if image is None:
    print("Error: Image not found. Check the file path.")
else:
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Display original and blurred images side by side
    print("Original Image:")
    cv2_imshow(image)
    print("Blurred Image:")
    cv2_imshow(blurred)

    # Optional: Save blurred image
    cv2.imwrite('/content/blurred_output.png', blurred)
