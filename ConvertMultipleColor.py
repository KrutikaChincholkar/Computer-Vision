import cv2
from google.colab.patches import cv2_imshow

# Step 1: Load image
image = cv2.imread("img.jpg")

# Step 2: Convert to different color spaces
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # Grayscale
hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    # Hue, Saturation, Value
lab  = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)    # Lightness, A*, B* (perceptual color model)
hls  = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)    # Hue, Lightness, Saturation

# Step 3: Display the images in Google Colab
cv2_imshow(image)  # Original BGR image
cv2_imshow(gray)   # Grayscale version
cv2_imshow(hsv)    # HSV color space
cv2_imshow(lab)    # LAB color space
cv2_imshow(hls)    # HLS color space
