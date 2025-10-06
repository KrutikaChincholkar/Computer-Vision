# Applies an average filter (normalized box filter)

import cv2
from google.colab.patches import cv2_imshow

# Step 1: Load the image
# 0 = grayscale, 1 = color
image = cv2.imread('/content/minion.png', 0)

# Check if the image loaded correctly
if image is None:
    print("Error: Image not found. Check the file path.")
else:
    print("Original Image:")
    cv2_imshow(image)

    # Step 2: Apply mean/box (average) filtering
    # The larger the kernel, the stronger the blur
    blurred_image = cv2.blur(image, (5, 5))

    # Step 3: Display the result
    print("Blurred Image (Mean Filter):")
    cv2_imshow(blurred_image)
