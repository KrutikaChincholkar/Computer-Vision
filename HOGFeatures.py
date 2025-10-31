# Step 1: Import Libraries
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Step 2: Load the Image
image = cv2.imread("/content/Waffle.jpg")  # Read the input image
cv2_imshow(image)  # Display the original image

# Step 3: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 4: Initialize the HOG Descriptor
hog = cv2.HOGDescriptor()

# Step 5: Compute the HOG features
hog_features = hog.compute(gray)

# Step 6: Print the shape and a small sample of HOG features
print("HOG Feature Vector Shape:", hog_features.shape)
print("Sample HOG feature values:\n", hog_features[:10].flatten())
