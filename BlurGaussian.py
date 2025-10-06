import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Read the image
image = cv2.imread('/content/Screenshot 2024-03-19 144347.png')

# Check if image loaded successfully
if image is None:
    print("Error: Image not found. Check your file path.")
else:
    # Apply Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(image, 100, 200)

    # Sharpen the image
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)

    # Display all images
    print("Original Image:")
    cv2_imshow(image)

    print("Gaussian Blur:")
    cv2_imshow(gaussian_blur)

    print("Canny Edge Detection:")
    cv2_imshow(edges)

    print("Sharpened Image:")
    cv2_imshow(sharpened)

    # (Optional) Save output images
    cv2.imwrite('/content/gaussian_blur.png', gaussian_blur)
    cv2.imwrite('/content/edges.png', edges)
    cv2.imwrite('/content/sharpened.png', sharpened)
